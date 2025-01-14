import numpy as np
from visualization import visualize_batch_with_labels
import time
import pickle
from tabulate import tabulate

class NetworkBuilder:
    def __init__(self, layers):
        '''
            param:
            layers - [class] - list of layers to implement in the model
        '''
        self.model = layers

    def forward(self, inputImages):
        output = inputImages
        for layer in self.model:
            output = layer.forward(output)
        prediction = np.argmax(output, axis = 1)
        return output, prediction

    def backward(self, dOutput, learning_rate):
        for i in range(1, len(self.model)+1):
            dOutput = self.model[-i].backward(dOutput, learning_rate)

    def test(self, X, y):
        success = 0
        set_size = X.shape[0]
        for i in range(set_size):
            _, prediction = self.forward(X[i])
            if prediction == y[i]:
                success +=1
        accuracy = success / set_size
        return accuracy

    def fit(self, X, y, learning_rate, lossFunction, epoch, batch_size):
        skipper = False
        if self.model[-1].activation_function == 'softmax' and lossFunction == 'cross_entropy_loss':
            self.model[-1].skip_softmax_in_back = True
            skipper = True
        else:
            self.model[-1].skip_softmax_in_back = False

        for e in range(epoch):
            batches_X, batches_y = self.build_batch(X, y, batch_size)
            batch_number = len(batches_X)

            for batch_idx in range(batch_number):
                outputs, _ = self.forward(batches_X[batch_idx])
                batch_exact_size = len(batches_y[batch_idx]) 
                nb_class = outputs.shape[1]
                goals = np.zeros((batch_exact_size, nb_class))
                for i in range(batch_exact_size):
                    goals[i, batches_y[batch_idx][i]] = 1
                
                dLoss, error = self.compute_loss(outputs, goals,lossFunction, batch_exact_size, skipper)

                self.backward(dLoss, learning_rate)

    def build_batch(self, X, y, batch_size):
        if batch_size > 1:
            set_size = X.shape[0]
            batch_number = set_size // batch_size

            combined = list(zip(X, y))
            np.random.shuffle(combined)
            shuffled_X, shuffled_y = zip(*combined)

            batches_X = [np.array(shuffled_X[i:i + batch_size]) for i in range(0, len(shuffled_X), batch_size)]
            batches_y = [np.array(shuffled_y[i:i + batch_size]) for i in range(0, len(shuffled_y), batch_size)]
        else:
            batches_X, batches_y = [np.array(X)], [np.array(y)]

        return batches_X, batches_y

    def compute_loss(self, outputs, goals, lossFunction, batchSize, skipper):
        match lossFunction:
            case 'cross_entropy_loss':
                error = (-1/batchSize)*np.sum(goals*np.log(outputs))
                if skipper:
                    dLoss = outputs - goals
                else:
                    dLoss = - goals / outputs
            case 'mse':
                error = (1/batch_size)*0.5*np.sum((outputs - goals)**2)
                dLoss = outputs - goals
        print(error)
        return dLoss, error


    def test(self, X, y):
        success = 0
        set_size = X.shape[0]
        _, prediction = self.forward(X)
        for i in range(len(prediction)):
            if prediction[i] == y[i]:
                success += 1
        accuracy = success / set_size
        return accuracy

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, filepath):
          with open(filepath, 'rb') as f:
            self.model = pickle.load(f)

    def print(self):
        table = []
        total_param = 0
        for i, layer in enumerate(self.model):
            training_weights = layer.weights.size + layer.biases.size
            total_param +=training_weights
            training_weights = '- -' if training_weights == 0 else training_weights
            trainable = True if training_weights != 0 else '- -'
            input_shape = list(layer.input_shape)
            input_shape[0] = 1
            output_shape = list(layer.output_shape)
            output_shape[0] = 1
            activation = layer.activation_function
            activation = '- -' if activation == None else activation
            table.append([layer, input_shape, output_shape, training_weights, trainable, activation])
        print(tabulate(table, headers=["Layer", "Input Shape", "Output Shape", "Param #", "Trainable", "Activation"], tablefmt="pretty", stralign="left", numalign="left"))
        print("total weights #:   ", total_param)
        

