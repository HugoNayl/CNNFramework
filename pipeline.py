from reader import read_idx1_ubyte, read_idx3_ubyte

import numpy as np

class Pipeline:
    '''
    Pipeline for data (preprocess include)
    Params:
    datasetPath - str: Path to files
    -- Files (x4) - str (x4): im train, label train, im test, label test
    '''
    def __init__(self, datasetPath, imageTrainFile, labelTrainFile, imageTestFile, labelTestFile):
        self.data_path = datasetPath
        self.image_train_file = imageTrainFile
        self.label_train_file = labelTrainFile
        self.image_test_file = imageTestFile
        self.label_test_file = labelTestFile

        self.images_train = None
        self.labels_train = None
        self.images_test = None
        self.labels_test = None

        self.import_files()
        self.images = [self.images_train, self.images_test]

        self.preprocess()

    def import_files(self):
        self.images_train = read_idx3_ubyte(self.data_path + '/' + self.image_train_file)
        self.labels_train = read_idx1_ubyte(self.data_path + '/' + self.label_train_file)
        self.images_test = read_idx3_ubyte(self.data_path + '/' + self.image_test_file)
        self.labels_test = read_idx1_ubyte(self.data_path + '/' + self.label_test_file)

    def normalize_to_range(self, data, new_min, new_max):
        old_min = 0
        old_max = 255
        
        normalized_data = (data - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
        return normalized_data

    def preprocess(self):
        resized_images = [[],[]]
        idx = 0
        for images in self.images:
            for i  in range(len(images)):
                im = np.pad(images[i], ((2, 2), (2, 2)), mode='constant').reshape(1,32,32)
                im = self.normalize_to_range(im, -0.1, 1.175).astype('float32')
                resized_images[idx].append(im)
            idx+=1

        self.images_train = np.array(resized_images[0])
        self.images_test = np.array(resized_images[1])

        print('images_train: ', self.images_train.shape)
        unique_classes, counts = np.unique(self.labels_train, return_counts=True)
        class_counts = dict(zip(unique_classes.tolist(), counts.tolist()))
        print(class_counts)

        print('images_test: ', self.images_test.shape)
        unique_classes, counts = np.unique(self.labels_test, return_counts=True)
        class_counts = dict(zip(unique_classes.tolist(), counts.tolist()))
        print(class_counts)


