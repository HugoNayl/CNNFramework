import numpy as np

from convolution import Convolution
from pooling import Pooling
from perceptronLayer import PerceptronLayer
from networkBuilder import NetworkBuilder
from pipeline import Pipeline
from conv2Perceptron import Conv2Perceptron

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
np.random.seed(42)
c3_connection_map = [ #test for partial connection but slow down a LOT
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 0],
        [5, 0, 1],
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 0],
        [4, 5, 0, 1],
        [5, 0, 1, 2],
        [0, 1, 3, 4],
        [1, 2, 4, 5],
        [0, 2, 3, 5],
        [0, 1, 2, 3, 4, 5]
        ]

layers = [
        Convolution([1, 32, 32],6, 5, 0, 1),
        Pooling('linear_sum', [2,2], 2, 'sigmoid', 6),
        Convolution([6, 14, 14], 16, 5, 0, 1),
        Pooling('linear_sum', [2,2], 2, 'sigmoid', 16),
        Convolution([16, 5, 5], 120, 5, 0, 1),
        Conv2Perceptron(),
        PerceptronLayer(120, 84, 'scaledTanh'),
        PerceptronLayer(84, 10, 'softmax')
        ]

model = NetworkBuilder(layers)
pipe = Pipeline('dataset', 'train-images.idx3-ubyte', 'train-labels.idx1-ubyte', 't10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')
train_im = pipe.images_train
train_lab = pipe.labels_train
test_im = pipe.images_test
test_lab = pipe.labels_test

acc_init = model.test(test_im, test_lab)

#Use model.load('test.pkl') to use model I trained
model.fit(train_im, train_lab, 0.001, 'cross_entropy_loss', 10, 128)

acc_end = model.test(test_im, test_lab)

print('accuracy: ', acc_init)
print('accuracy: ', acc_end)
model.print()
model.save_model('test2.pkl')

