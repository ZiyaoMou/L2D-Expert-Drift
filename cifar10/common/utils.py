import torch
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    
import random

def metrics_print_baseline(net_class,   expert_fn, n_classes, loader):
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs_class = net_class(images)
            _, predicted = torch.max(outputs_class.data, 1)
            batch_size = outputs_class.size()[0]            # batch_size
            
            exp_prediction = expert_fn(images, labels)
            for i in range(0,batch_size):
                r = (exp_prediction[i] == labels[i].item())
                if r==0:
                    total += 1
                    prediction = predicted[i]
                    if predicted[i] == 10:
                        max_idx = 0
                        for j in range(0,10):
                            if outputs_class.data[i][j] >= outputs_class.data[i][max_idx]:
                                max_idx = j
                        prediction = max_idx
                    else:
                        prediction = predicted[i]
                    correct += (prediction == labels[i]).item()
                    correct_sys += (prediction == labels[i]).item()
                if r==1:
                    exp += (exp_prediction[i] == labels[i].item())
                    correct_sys +=(exp_prediction[i] == labels[i].item())
                    exp_total+=1
                real_total += 1
    cov = str(total) + str(" out of") + str(real_total)
    to_print={"coverage":cov, "system accuracy": 100*correct_sys/real_total, "expert accuracy":100* exp/(exp_total+0.0002),"classifier accuracy":100*correct/(total+0.0001) }
    print(to_print)


def metrics_print(net, expert_fn, n_classes, loader):
    net = net.to(device)
    net.eval()

    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    alone_correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = net(images)   # now inputs and weights are on same device
            _, predicted = torch.max(outputs, 1)

            batch_size = outputs.size(0)
            exp_prediction = expert_fn(images, labels)  # if expert_fn only uses labels, can put on CPU; if uses images, need device consistency

            for i in range(batch_size):
                r = (predicted[i].item() == n_classes)  # n_classes as "reject class" index
                prediction = predicted[i]
                if predicted[i] == 10:  # note: if n_classes is not 10, this should be changed to n_classes
                    max_idx = 0
                    for j in range(0, 10):  # same as above, recommend using n_classes instead of hardcoded 10
                        if outputs[i][j] >= outputs[i][max_idx]:
                            max_idx = j
                    prediction = max_idx
                else:
                    prediction = predicted[i]

                alone_correct += (prediction == labels[i]).item()
                if r == 0:
                    total += 1
                    correct += (predicted[i] == labels[i]).item()
                    correct_sys += (predicted[i] == labels[i]).item()
                else:
                    exp += (exp_prediction[i] == labels[i].item())
                    correct_sys += (exp_prediction[i] == labels[i].item())
                    exp_total += 1
                real_total += 1

    cov = f"{total} out of {real_total}"
    to_print = {
        "coverage": cov,
        "system accuracy": 100 * correct_sys / real_total,
        "expert accuracy": 100 * exp / (exp_total + 1e-4),
        "classifier accuracy": 100 * correct / (total + 1e-4),
        "alone classifier": 100 * alone_correct / real_total
    }
    print(to_print)