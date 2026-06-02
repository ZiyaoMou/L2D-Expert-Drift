import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import sys
import numpy as np
import torch.backends.cudnn as cudnn
from evaluate.evaluate_ce import test_model
import torchvision

use_gpu = torch.cuda.is_available()

class DenseNet121_CE(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """
    def __init__(self, out_size):
        super(DenseNet121_CE, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size * 2) # for CE use 2, for LCE use 3
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


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


class CheXpertTrainer_CE():
   
    def train(model, dataLoaderTrain, dataLoaderVal, dataLoaderTest,nnClassCount, trMaxEpoch, launchTimestamp, checkpoint):
        
        #SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, mode='min')

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
            #test_model(model, dataLoaderVal )
            batchs, losst, losse = CheXpertTrainer_CE.epochTrain(model, dataLoaderTrain, dataLoaderTest, optimizer, trMaxEpoch, nnClassCount, loss)
            lossVal = CheXpertTrainer_CE.epochVal(model, dataLoaderVal, optimizer, trMaxEpoch, nnClassCount, loss)
            test_model(model, dataLoaderVal)

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            scheduler.step(lossVal)

            if lossVal < lossMIN:
                lossMIN = lossVal    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'm-epoch'+str(epochID)+'-' + launchTimestamp + '_min' + '.pth.tar')
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
            else:
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossVal, 'optimizer' : optimizer.state_dict()}, 'm-epoch'+str(epochID)+'-' + launchTimestamp + '.pth.tar')
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
        
        return batchs, losst, losse        

    def epochTrain(model, dataLoader, testDataLoader, optimizer, trMaxEpoch, nnClassCount, loss):
        
        batch = []
        losstrain = []
        losseval = []
        
        model.train()

        for batchID, (varInput, target, rad1, rad2, rad3, weights) in enumerate(dataLoader):
            varTarget = target.cuda(non_blocking = True)
            varOutput = model(varInput)
            lossvalue = CrossEntropyLoss(varOutput, target,  weights.cuda(non_blocking=True) )
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            
            l = lossvalue.item()
            losstrain.append(l)
            
            if batchID%2800==0:
                print(batchID//280, "% batches computed")
                #Fill three arrays to see the evolution of the loss
                batch.append(batchID)
                le = CheXpertTrainer_CE.epochVal(model, testDataLoader, optimizer, trMaxEpoch, nnClassCount, loss).item()
                losseval.append(le)
                print(batchID)
                print(l)
                print(le)
                
        return batch, losstrain, losseval

    def epochVal(model, dataLoader, optimizer, epochMax, classCount, loss):
            model.eval()
            lossVal = 0
            lossValNorm = 0

            with torch.no_grad():
                for i, (varInput, target, rad1, rad2, rad3, weights) in enumerate(dataLoader):
                    target = target.cuda(non_blocking = True)
                    varOutput = model(varInput)
                    losstensor = CrossEntropyLoss(varOutput, target,  weights.cuda(non_blocking=True) )
                    lossVal += losstensor
                    lossValNorm += 1
                    
            outLoss = lossVal / lossValNorm
            return outLoss
    
  