from Loss_Functions.Loss import Loss
import numpy as np
class Mean_SE(Loss):
    def __init__ (self, prediction, target):
        self.prediction = prediction
        self.target = target
        self.loss = np.mean((prediction- target)**2)
        return self.loss
    def backward (self):
        gradient = 2 * (self.prediction - self.target) / self.target.size
        return gradient