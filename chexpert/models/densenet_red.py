import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import sys
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision

use_gpu = torch.cuda.is_available()
def CrossEntropyLoss(outputs, labels, weights):
    # m: expert costs, labels: ground truth, n_classes: number of classes
    '''
    outputs: model outputs
    labels: target vector
    weights: uncertainty flagging
    '''
    batch_size = outputs.size()[0]
    total_loss = 0
    j = 0
    for i in range(14):
        out_softmax = torch.nn.functional.softmax(outputs[range(batch_size),j:j+2])
        j += 2
        loss_class = - weights[range(batch_size),i] * torch.log2(out_softmax[range(batch_size), labels[range(batch_size),i].long()])   
        loss_class = torch.sum(loss_class)/(torch.sum(weights[range(batch_size),i] ) +0.000000001)
        total_loss += loss_class
    total_loss /= 14
    return total_loss


class CheXpertTrainer_rad():
   
    def train(model, rad_index, dataLoaderTrain, dataLoaderTest, nnClassCount, trMaxEpoch, launchTimestamp, checkpoint):
        
        #SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=2, mode='min')

        #SETTINGS: LOSS
        loss = 0
        #LOAD CHECKPOINT 
        if checkpoint != None and use_gpu:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
        
        #TRAIN THE NETWORK
        lossMIN = 100000
        
        for epochID in range(0, trMaxEpoch):
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
            batchs, losst, losse = CheXpertTrainer_rad.epochTrain(model, rad_index, dataLoaderTrain, dataLoaderTest, optimizer, trMaxEpoch, nnClassCount, loss)
            lossVal = 0
            #lossVal = CheXpertTrainer_rad.epochVal(model, rad_index, dataLoaderVal, optimizer, trMaxEpoch, nnClassCount, loss)
            #test_model_rad(model, dataLoaderVal, 1)

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            scheduler.step(lossVal)

            if lossVal < lossMIN:
                lossMIN = lossVal    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'rad_model-epoch'+str(epochID)+'-' + launchTimestamp + '_min' + '.pth.tar')
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
            else:
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossVal, 'optimizer' : optimizer.state_dict()}, 'rad_model-epoch'+str(epochID)+'-' + launchTimestamp + '.pth.tar')
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
        
        return batchs, losst, losse        
    #-------------------------------------------------------------------------------- 



    def epochTrain(model, rad_index, dataLoaderTrain, dataLoaderTest, optimizer, epochMax, classCount, loss):
        
        batch = []
        losstrain = []
        losseval = []
        
        model.train()

        for batchID, (varInput, target, rad1, rad2, rad3, weights) in enumerate(dataLoaderTrain):
            
            varTarget = 0 
            if rad_index == 1:
                varTarget = (rad1.cuda(non_blocking = True) == target.cuda(non_blocking = True) ) *1.0
            elif rad_index == 2:
                varTarget = (rad2.cuda(non_blocking = True)== target.cuda(non_blocking = True) ) *1.0
            elif rand_index == 3:
                varTarget = (rad3.cuda(non_blocking = True)== target.cuda(non_blocking = True) ) *1.0


            varOutput = model(varInput)
            lossvalue = CrossEntropyLoss(varOutput, varTarget,  weights.cuda(non_blocking=True) )
                       
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            
            l = lossvalue.item()
            losstrain.append(l)
            
            if batchID%2800==0:
                print(batchID//280, "% batches computed")
                #Fill three arrays to see the evolution of the loss


                batch.append(batchID)
                
                le = CheXpertTrainer_rad.epochVal(model, rad_index, dataLoaderTest, optimizer, epochMax, classCount, loss).item()
                losseval.append(le)
                
                print(batchID)
                print(l)
                print(le)
                
        return batch, losstrain, losseval
    

    #-------------------------------------------------------------------------------- 


    def epochVal(model, rad_index, dataLoader, optimizer, epochMax, classCount, loss):
            
            model.eval()
            
            lossVal = 0
            lossValNorm = 0

            with torch.no_grad():
                for i, (varInput, target, rad1, rad2, rad3, weights) in enumerate(dataLoader):
                    
                    varTarget = 0 
                    if rad_index == 1:
                        varTarget = (rad1.cuda(non_blocking = True) == target.cuda(non_blocking = True) ) *1.0
                    elif rad_index == 2:
                        varTarget = (rad2.cuda(non_blocking = True)== target.cuda(non_blocking = True) ) *1.0
                    elif rand_index == 3:
                        varTarget =( rad3.cuda(non_blocking = True)== target.cuda(non_blocking = True) ) *1.0
                    varOutput = model(varInput)
                    
                    losstensor = CrossEntropyLoss(varOutput, varTarget,  weights.cuda(non_blocking=True) )
                    lossVal += losstensor
                    lossValNorm += 1
                    
            outLoss = lossVal / lossValNorm
            return outLoss
    
  
    
    #--------------------------------------------------------------------------------     
     
    #---- Computes area under ROC curve 
    #---- dataGT - ground truth data
    #---- dataPRED - predicted data
    #---- classCount - number of classes
    
    def computeAUROC (dataGT, dataPRED, classCount):
        
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            try:
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except ValueError:
                pass
        return outAUROC
        
        
    #-------------------------------------------------------------------------------- 
    
    
    def test(model, rad_index, dataLoaderTest, nnClassCount, checkpoint, class_names):   
        
        cudnn.benchmark = True
        
        if checkpoint != None and use_gpu:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])

        if use_gpu:
            outGT = torch.FloatTensor().cuda()
            outPRED = torch.FloatTensor().cuda()
        else:
            outGT = torch.FloatTensor()
            outPRED = torch.FloatTensor()
       
        model.eval()
        
        with torch.no_grad():
            for i, (input, target, rad1, rad2, rad3, weights) in enumerate(dataLoaderTest):

                vartarget = 0 
                if rad_index == 1:
                    vartarget = (rad1.cuda(non_blocking = True) == target.cuda(non_blocking = True) ) *1.0
                elif rad_index == 2:
                    vartarget = (rad2.cuda(non_blocking = True)== target.cuda(non_blocking = True) ) *1.0
                elif rad_index == 3:
                    vartarget =( rad3.cuda(non_blocking = True)== target.cuda(non_blocking = True) ) *1.0
                outGT = torch.cat((outGT, vartarget), 0).cuda()

                bs, c, h, w = input.size()
                varInput = input.view(-1, c, h, w)
            
                out = model(varInput)
                batch_size = out.size()[0]  # batch_size
                j = 0
                output = torch.FloatTensor([]).cuda()
                for k in range(14):
                    out_softmax = torch.nn.functional.softmax(out[range(batch_size),j:j+2])
                    output = torch.cat((output,torch.FloatTensor([out_softmax[0][1].detach()]).cuda()),0)
                    j += 2
                #out = out[range(batch_size),range(0,nnClassCount*2)[1%2::2]] # get only ones
                #print(out[range(batch_size),range(0,nnClassCount*2)[1%2::2]])
                outPRED = torch.cat((outPRED, output.view(1, -1)), 0)

        aurocIndividual = CheXpertTrainer_rad.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        
        print ('AUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print (class_names[i], ' ', aurocIndividual[i])
        
        return outGT, outPRED