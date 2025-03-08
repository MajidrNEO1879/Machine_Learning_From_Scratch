import numpy as np
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def __call__(self, input: np.ndarray) -> np.ndarray:
        return self.forward(input)

    def forward(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, up_gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def steps(self, learning_rate: float) -> None:
        pass
