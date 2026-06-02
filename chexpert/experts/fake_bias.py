import numpy as np

class ExpertModelBiased:
    def __init__(self, confounding_class, seq_len, num_classes,
                 p_confound=0.7, p_nonconfound=1.0,
                 decay_confound=0.05, decay_nonconfound=0.0,
                 use_fatigue=True):
        """
        seq_len         : total number of timesteps
        num_classes     : number of prediction classes (K)
        p_confound      : base accuracy for confound class
        p_nonconfound   : base accuracy for non-confound class
        decay_confound  : decay per timestep for confound
        decay_nonconfound: decay per timestep for non-confound
        use_fatigue     : whether to apply time-based decay
        """
        self.seq_len = seq_len
        self.K = num_classes
        self.p_confound = p_confound
        self.p_nonconfound = p_nonconfound
        self.decay_confound = decay_confound
        self.decay_nonconfound = decay_nonconfound
        self.use_fatigue = use_fatigue
        self.confounding_class = confounding_class

    def predict(self, y_batch, timestep):
        """
        y_batch           : numpy array [B, K] binary ground truth labels
        confound_indicator: numpy array [B], 1 if confounded, 0 if not
        timestep          : int, current timestep (starts at 0)
        returns           : numpy array [B, K] binary predictions
        """
        B, K = y_batch.shape
        acc_matrix = np.zeros((B, K))

        for i in range(B):
            for k in range(K):
                if y_batch[i][self.confounding_class] == 1:
                    acc = self.p_confound - self.decay_confound * timestep if self.use_fatigue else self.p_confound
                else:
                    acc = self.p_nonconfound - self.decay_nonconfound * timestep if self.use_fatigue else self.p_nonconfound
                acc_matrix[i, k] = max(0.0001, acc)

        rand = np.random.rand(B, K)
        flip_mask = rand > acc_matrix
        pred = np.where(flip_mask, 1 - y_batch, y_batch)
        return pred