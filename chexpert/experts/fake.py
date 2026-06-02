import numpy as np

class ExpertModel_fake():
    def __init__(self, confounding_class, p_confound, p_nonconfound):
        self.confounding_class = confounding_class
        self.p_confound = p_confound
        self.p_nonconfound = p_nonconfound
    def predict(self, y):
        '''
        Using backoff
        y: list of targets where each is a python list of size 14
        returns: expert predictions for each of  expert 1
        '''
        preds = [[] for _ in range(3)]
        for rad_index in range(3):
            for i in range(len(y)):
                pred = [0] *14
                key_all = str(y[i])
                for cls in range(14):
                    if y[i][self.confounding_class] == 0:
                        if y[i][cls] == 1:
                            prediction = np.random.binomial(1,self.p_confound,1)[0]
                            pred[cls] = prediction
                        else:
                            prediction = np.random.binomial(1,1- self.p_confound,1)[0]
                            pred[cls] = prediction
                    else:
                        if y[i][cls] == 1:
                            prediction = np.random.binomial(1,self.p_nonconfound,1)[0]
                            pred[cls] = prediction
                        else:
                            prediction = np.random.binomial(1,1- self.p_nonconfound,1)[0]
                            pred[cls] = prediction       
                preds[rad_index].append(pred)
        return preds

