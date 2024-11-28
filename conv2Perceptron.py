import numpy as np

class Conv2Perceptron:
    '''
    Class for adapt convolution layer to fully connected layer
    '''
    def __init__(self):
        self.input_shape = None
        self.weights = np.array([[]])
        self.biases = np.array([[]])
        self.activation_function = None
        self.input_shape = None
        self.output_shape = None

    def __str__(self):
        return 'Conv2Perceptron'

    def forward(self, inputMatrix):
        self.input_shape = inputMatrix.shape
        batch_size = self.input_shape[0]
        output = inputMatrix.reshape(batch_size, -1)
        self.output_shape = output.shape
        return output

    def backward(self, dLoss, learning_rate):
        return dLoss.reshape(self.input_shape)

if __name__ == "__main__":
    c2p = Conv2Perceptron()
    im = np.array([[[[1,2,3,4],
                     [5,6,7,8],
                     [9,10,11,12],
                     [13,14,15,16]]],
                   [[[1,2,3,4],
                     [5,6,7,8],
                     [9,10,11,12],
                     [13,14,15,16]]]])

    out = c2p.forward(im)
    back = c2p.backward(out, 1)
    print(out)
    print(back)
