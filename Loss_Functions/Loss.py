import numpy as np
class Loss:
    def __init__ (self):
        self.prediction = None
        self.target = None
        self.loss = None
    def __calls__ (self, prediction, target):
        return self.forward(prediction, target)
    def forward (seld, prediction, target):
        raise NotImplementedError
    def backward (self):
        raise NotImplementedError
