#df/dx in sigmoid => f(x)*(1-f(x))
from Activation_Functions.Layer import Layer
import numpy as np
class Sigmoid(Layer):
    def forward (self, input):
        self.output = 1/(1  + np.exp(-input))
        return self.output
    def backward(self, up_gradient):
        down_gradient = self.output*(1-self.output) * up_gradient
        return down_gradient