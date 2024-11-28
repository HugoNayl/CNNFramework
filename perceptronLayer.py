import numpy as np

class PerceptronLayer:
    def __init__(self, inputSize, nbPerceptron, activationFunction):
        '''
        param:
        inputSize - int: size of input vector (number of features)
        nbPerceptron - int: number of perceptron in the layer
        activationFunction - str: relu | sigmoid | softmax | tanh | ScaledTanh
        '''
        self.n = inputSize
        self.m = nbPerceptron
        self.activation_function = activationFunction
        self.skip_softmax_in_back = False

        limit = np.sqrt(6 / (self.n + self.m))
        self.weights = np.random.uniform(-limit, limit, (self.m, self.n))
        self.biases = np.zeros((self.m))

        self.weights_velocities = np.zeros_like(self.weights)
        self.biases_velocities = np.zeros_like(self.biases)

        self.input = None
        self.output = None
        self.input_shape = None
        self.output_shape = None

        self.momentum = 0.9

    def __str__(self):
        return "Fully Connected"

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def softmax(self, x):
        batch_size = x.shape[0]
        e_x = np.exp(x - np.max(x)) #numerical stability for little and big numbers
        return e_x/np.sum(e_x, axis = 1).reshape(batch_size, -1)

    def tanh(self, x):
        return np.tanh(x)

    def atanh(self, x):
        return 1.7159 * np.tanh(x)

    def forward(self, inputVector):
        self.input_shape = inputVector.shape
        self.input = inputVector
        output_vector = np.dot(self.input, self.weights.T) + self.biases.T[np.newaxis, :]
        match self.activation_function:
            case 'relu':
                self.output = self.relu(output_vector)
            case 'sigmoid':
                self.output = self.sigmoid(output_vector)
            case 'softmax':
                self.output = self.softmax(output_vector)
            case 'tanh':
                self.output = self.tanh(output_vector)
            case 'scaledTanh':
                self.output = 1.7159 * self.tanh(2/3*output_vector)
        self.output_shape = self.output.shape
        return self.output

    def backward(self, dLossa, learning_rate):
        '''
        dLossa: gradient of loss w.r.t a (loss of previous layer(s))
        dLoss: gradient of loss w.r.t z (loss with activation function loss computed)
        fLossam1: activation function of layer - 1 (Loss given to next layer, layer -1 activation_function) 
        '''
        if self.skip_softmax_in_back == False:
            match self.activation_function:
                case 'relu':
                    dA = (self.output > 0).astype(int)
                case 'sigmoid':
                    dA = self.output * (1 - self.output)
                case 'softmax':
                    dA = np.diag(self.output) - np.outer(self.output, self.output.T)
                case 'tanh':
                    dA = 1-np.square(self.output)
                case 'scaledTanh':
                    dA = 1.7159*2/3*(1-np.square(1/1.7159* self.output))
            dLoss = dLossa * dA
        else:
            dLoss = dLossa

        dLossam1 = np.dot(dLoss, self.weights)

        avg_dLoss = dLoss/dLoss.shape[0] #prepare for averaging in dot product

        dWeights = np.dot(avg_dLoss.T, self.input)
        dBiases = np.sum(avg_dLoss, axis = 0)

        self.weights_velocities = self.momentum * self.weights_velocities - learning_rate * dWeights
        self.biases_velocities = self.momentum * self.biases_velocities - learning_rate * dBiases

        self.weights += self.weights_velocities
        self.biases += self.biases_velocities

        return dLossam1

if __name__ == "__main__":
    mlp = PerceptronLayer(5, 2, 'relu')
    inputVector = np.array([[1,2,3,4,5],[6,7,8,9,10]])
    mlp.weights = np.ones((2, 5))
    mlp.biases = np.zeros((2))
    output = mlp.forward(inputVector)
    print(output)
    dLoss = np.ones_like(output)
    back = mlp.backward(dLoss, 0.1)
    print(mlp.weights)
    print(mlp.biases)
    print(back)
