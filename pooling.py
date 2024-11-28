import numpy as np

class Pooling:
    def __init__(self, poolingType, poolingSize, stride, activationFunction = None, layer = None):
        '''
        params:
        poolingType - sting: avg | max | linear_sum
        poolingSize - [int, int]: [height, weidth]
        stride - int: stride NOT IMPLEMENTED YET (=2)
        activationFunction - str: tanh | sigmoid | None
        '''
        self.type = poolingType
        self.poolingSize = poolingSize
        self.stride = stride
        self.activation_function = activationFunction

        self.weights = np.array([[]])
        self.biases = np.array([[]])

        if poolingType == 'linear_sum':
            limit = np.sqrt(6/layer)
            self.weights = np.random.uniform(-limit, limit, (layer,1,1))
            self.biases = np.zeros((layer,1,1))
            self.weights_velocities = np.zeros_like(self.weights)
            self.biases_velocities = np.zeros_like(self.biases)
            self.momentum = 0.9


        self.input = None
        self.output = None
        self.input_shape = None
        self.output_shape = None

    def __str__(self):
        return 'Pooling'

    def forward(self, image):
        self.input_shape = image.shape
        self.input = image
        ph, pw = self.poolingSize
        batch_size, l, h, w = image.shape
        output_height = h // ph
        output_width = w // pw
        image_reshaped = image.reshape(batch_size, l, output_height, ph, output_width, pw)
            
        if self.type == 'avg':
            self.output = image_reshaped.mean(axis=(3, 5))
        elif self.type == 'max':
            self.output = image_reshaped.max(axis=(3, 5))
        elif self.type == 'linear_sum':
            self.input_reshaped = image_reshaped.sum(axis=(3, 5))
            self.output = self.input_reshaped*self.weights + self.biases
        
        self.output_shape = self.output.shape
        match self.activation_function:
            case 'tanh':
                self.output = np.tanh(self.output)
                return self.output
            case 'sigmoid':
                result = np.empty_like(self.output)
                pos_mask = self.output >= 0
                neg_mask = ~pos_mask  # x < 0
                # For positive x, use the standard formula
                result[pos_mask] = 1 / (1 + np.exp(-self.output[pos_mask]))
                # For negative x, reformulate to avoid overflow
                exp_x = np.exp(self.output[neg_mask])
                result[neg_mask] = exp_x / (1 + exp_x)
                self.output = 1.0/(1.0 + np.exp(-self.output))
                return self.output
            case None:
                return self.output

    def backward(self, dLossa, learning_rate):
        match self.activation_function:
            case 'tanh':
                dA = 1-np.square(self.output)
            case 'sigmoid':
                dA = self.output * (1 - self.output)
            case None:
                dA = np.ones_like(self.output)
        dLoss = dLossa * dA

        batch_size, l, h, w = self.input.shape
        ph, pw = self.poolingSize
        

        match self.type:
            case 'avg':
                px_in_pooling = ph * pw
                dLoss = dLoss * 1/px_in_pooling
                dLossalm1 = np.repeat(np.repeat(dLoss, ph, axis=2), pw, axis = 3)
            case 'max':
                output_height = h // ph
                output_width = w // pw
                dLoss = np.repeat(np.repeat(dLoss, ph, axis=2), pw, axis = 3)
                dLoss_reshaped = dLoss.reshape(l, output_height, ph, output_width, pw)
                image_reshaped = self.input.reshape(l, output_height, ph, output_width, pw)
                max_values = image_reshaped.max(axis=(2, 4), keepdims=True)
                binary_mask = (image_reshaped == max_values).astype(int)
                applied_mask = binary_mask * dLoss_reshaped
                dLossalm1 = applied_mask.reshape(l, h, w)
            case 'linear_sum':

                dLoss = dLoss * self.weights
                dLossalm1 = np.repeat(np.repeat(dLoss, ph, axis=2), pw, axis = 3)


                avg_dLoss = dLoss/batch_size
                dWeights = avg_dLoss * self.input_reshaped
                dWeights = dWeights.sum(axis = (0, 2, 3)).reshape(l, 1, 1)
                dBiases = avg_dLoss.sum(axis = (0, 2, 3)).reshape(l, 1, 1)

                self.weights_velocities = self.momentum * self.weights_velocities - learning_rate * dWeights
                self.biases_velocities = self.momentum * self.biases_velocities - learning_rate * dBiases

                self.weights += self.weights_velocities
                self.biases += self.biases_velocities


        return dLossalm1

if __name__ == '__main__':

    input_data = np.array([[[[1, 3, 2, 4],
                             [5, 6, 1, 2],
                             [3, 2, 7, 8],
                             [1, 2, 3, 4]],
                            [[2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2]]
                            ],
                            [[[1, 1, 1, 1],
                             [1, 1, 1, 1],
                             [1, 1, 1, 1],
                             [1, 1, 1, 1]],
                            [[1, 1, 1, 1],
                             [1, 1, 1, 1],
                             [1, 1, 1, 1],
                             [1, 1, 1, 1]]]])  # Shape: (1, 4, 4)
 
    pooling_layer = Pooling(poolingSize=(2, 2), poolingType='linear_sum', stride = 2, activationFunction = None, layer = 2)
    print(input_data)
# Forward pass
    output = pooling_layer.forward(input_data)
    print("Forward Output:\n", output)


    # Assume gradient from next layer is ones
    dLossa = np.ones_like(output)

    # Backward pass
    dLoss_input = pooling_layer.backward(dLossa, learning_rate=0.01)
    print("Backward Gradient:\n", dLoss_input)
