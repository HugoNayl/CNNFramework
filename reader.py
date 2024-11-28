import struct
import numpy as np

def read_idx1_ubyte(file_path):
    with open(file_path, 'rb') as f:
        # Read the magic number (first 4 bytes)
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number != 2049:  # Magic number for IDX1 files
            raise ValueError("Invalid magic number. This is not an IDX1 file.")

        # Read the number of labels (next 4 bytes)
        num_items = struct.unpack('>I', f.read(4))[0]

        # Read the labels (remaining bytes)
        labels = []
        for _ in range(num_items):
            label = struct.unpack('>B', f.read(1))[0]
            labels.append(label)

        return labels

def read_idx3_ubyte(file_path):
    with open(file_path, 'rb') as f:
        # Read header information
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number != 2051:
            raise ValueError("Invalid magic number. This is not an IDX3 file.")
        
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_columns = struct.unpack('>I', f.read(4))[0]
        
        # Read image data
        image_size = num_rows * num_columns
        images = np.zeros((num_images, num_rows, num_columns), dtype=np.uint8)
        
        for i in range(num_images):
            image_data = f.read(image_size)
            images[i] = np.frombuffer(image_data, dtype=np.uint8).reshape(num_rows, num_columns)
        
        return images


if __name__ == '__main__':
    file_path = 'dataset/train-images.idx3-ubyte'
    images = read_idx3_ubyte(file_path)
    print("Shape of images:", images.shape)
    print("First image data:", images[0])
    file_path = 'dataset/train-labels.idx1-ubyte'
    labels = read_idx1_ubyte(file_path)
    print("First 10 labels:", labels[:1])
