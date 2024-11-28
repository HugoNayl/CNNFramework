import numpy as np
from numpy.lib.stride_tricks import as_strided
import time

class Convolution:
    '''
    convolution class 

    param:
    inputSize - [int, int, int]: [layer, height, width]
    nbKernel - int: number of kernel to create
    kernelSize - int: size of the kernel (kernelSize*kernelSize)
    padding - int: padding size
    stride - int: stride size
    activationFunction - str: tanh | None
    '''
    def __init__(self, inputSize, nbKernel, kernelSize, padding, stride, activationFunction = None):

        if inputSize[1] + 2 * padding < kernelSize or inputSize[2] + 2 * padding < kernelSize:
            raise Exception("kernel must be smaller than input feature map")

        self.input_size = inputSize
        self.nb_kernel = nbKernel
        self.kernel_size = kernelSize
        self.padding = padding
        self.stride = stride
        self.activation_function = activationFunction

        self.n_output = (padding * 2 + inputSize[2] - kernelSize)//stride + 1
        self.m_output = (padding * 2 + inputSize[1] - kernelSize)//stride + 1
        self.num_patch = self.n_output * self.m_output
        self.patch_size = kernelSize * kernelSize * inputSize[0]

        limit = np.sqrt(6 / (self.patch_size*nbKernel))
        self.weights = np.random.uniform(-limit, limit, (nbKernel, self.patch_size))
        self.biases = np.zeros((nbKernel))

        self.weights_velocities = np.zeros_like(self.weights)
        self.biases_velocities = np.zeros_like(self.biases)

        self.momentum = 0.9
        
        self.batch_size = None
        self.output = None
        self.input_shape = None
        self.patches = None
        self.output_shape = None

    def __str__(self):
        return 'Convolution'

    def im2patch(self, im):
        self.batch_size = im.shape[0]
        im_padded = np.pad(im, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        new_shape = (
            self.batch_size,
            self.input_size[0],
            self.m_output,
            self.n_output,
            self.kernel_size,
            self.kernel_size
        )
    
        new_strides = (
            im_padded.strides[0],
            im_padded.strides[1],
            im_padded.strides[2] * self.stride,
            im_padded.strides[3] * self.stride,
            im_padded.strides[2],
            im_padded.strides[3],
        )

        patches = as_strided(im_padded, shape=new_shape, strides=new_strides)

        patches = patches.reshape(
            self.batch_size,
            self.input_size[0],
            self.num_patch,
            self.kernel_size,
            self.kernel_size
        )
        patches = patches.transpose(0, 2, 1, 3, 4).reshape(self.batch_size * self.num_patch, self.patch_size)
        return patches

    def col2im(self, col):
        return col.reshape(self.batch_size, self.m_output, self.n_output, self.nb_kernel).transpose(0, 3, 1, 2)

    def forward(self, inputImage):
        self.input_shape = inputImage.shape
        self.patches = self.im2patch(inputImage)
        col_output = np.dot(self.patches, self.weights.T) + self.biases.T[np.newaxis, :]
        self.output = self.col2im(col_output)
        self.output_shape = self.output.shape
        match self.activation_function:
            case 'tanh':
                self.output = np.tanh(self.output)
                return self.output
            case None:
                return self.output

    def patch2im(self, patches):
        fm, h, w = self.input_size
        im_padded = np.zeros((self.batch_size, fm, h + 2 * self.padding, w + 2 * self.padding))

        patches = patches.reshape(self.batch_size, self.num_patch, self.patch_size)
        patches = patches.reshape(self.batch_size, self.num_patch, fm, self.kernel_size, self.kernel_size)
        
        patch_idx = 0
        for i in range(self.m_output):
            for j in range(self.n_output):
                i_start = i * self.stride
                j_start = j * self.stride
                im_padded[:, :, i_start:i_start + self.kernel_size, j_start:j_start + self.kernel_size] += patches[:, patch_idx]
                patch_idx += 1
        if self.padding > 0:
            im = im_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            im = im_padded
        return im

    def backward(self, dLossa, learning_rate):
        '''
        dLossa: gradient of loss w.r.t a (loss of previous layer(s))
        dLoss: gradient of loss w.r.t z (loss with activation function loss computed)
        fLossam1: activation function of layer - 1 (Loss given to next layer, layer -1 activation_function) 
        '''
        if self.activation_function == 'tanh':
            dA = 1 - np.square(self.output)
        else:
            dA = np.ones_like(dLossa)

        dLoss = dLossa * dA
        dLoss_row = dLoss.transpose(0, 2, 3, 1).reshape(self.batch_size * self.num_patch, self.nb_kernel)

        dLossalm1_patch = np.dot(dLoss_row, self.weights)
        dLossalm1 = self.patch2im(dLossalm1_patch)

        dWeights = np.dot(dLoss_row.T, self.patches)
        dBiases = np.sum(dLoss_row, axis=0)

        self.weights_velocities = self.momentum * self.weights_velocities - learning_rate * dWeights
        self.biases_velocities = self.momentum * self.biases_velocities - learning_rate * dBiases

        self.weights += self.weights_velocities
        self.biases += self.biases_velocities

        return dLossalm1

if __name__ == '__main__':
    im = np.array([[[[2, 1, 1, 4],
                     [2, 1, 1, 4],
                     [2, 1, 1, 4],
                     [2, 1, 1, 4]],
                    [[2, 2, 2, 4],
                     [2, 2, 2, 4],
                     [2, 2, 2, 4],
                     [2, 2, 2, 4]]],
                   [[[2, 3, 3, 4],
                     [2, 3, 3, 4],
                     [2, 3, 3, 4],
                     [2, 3, 3, 4]],
                    [[2, 4, 4, 4],
                     [2, 4, 4, 4],
                     [2, 4, 4, 4],
                     [2, 4, 4, 4]]]])
    convo = Convolution([2,4,4],2 ,3 ,0 ,1)
    convo.weights = np.ones((2, 18))
    convo.biases = np.zeros((2))
    output = convo.forward(im)
    print(output)
    dLoss = np.ones_like(output)
    back = convo.backward(dLoss, 0.1)
    print(convo.weights)
    print(convo.biases)
    print(back)

