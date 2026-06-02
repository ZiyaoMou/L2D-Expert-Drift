import random
n_dataset = 10  # cifar-10
k = 5

class synth_expert:
    '''
    simple class to describe our synthetic expert on CIFAR-10
    ----
    k: number of classes expert can predict
    n_classes: number of classes (10+1 for CIFAR-10)
    '''
    def __init__(self, k, n_classes):
        self.k = k
        self.n_classes = n_classes

    def predict(self, input, labels):
        batch_size = labels.size()[0]  # batch_size
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i].item() <= self.k:
                outs[i] = labels[i].item()
            else:
                prediction_rand = random.randint(0, self.n_classes - 1)
                outs[i] = prediction_rand
        return outs


expert = synth_expert(k, n_dataset)