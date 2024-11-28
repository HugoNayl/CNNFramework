import numpy as np
import matplotlib.pyplot as plt

def visualize_batch_with_labels(batch, labels, class_names=None, num_images=30, title='Batch Images with Labels'):
    """
    Visualize a batch of grayscale images with their labels.

    Parameters:
    - batch: NumPy array of shape (batch_size, 1, height, width)
    - labels: List or NumPy array of labels corresponding to each image
    - class_names: List of class names corresponding to label indices (optional)
    - num_images: Number of images to display
    - title: Title for the plot
    """
    batch_size = len(batch)
    num_images = min(num_images, batch_size)

    # Determine the grid size for plotting
    grid_cols = int(np.ceil(np.sqrt(num_images)))
    grid_rows = int(np.ceil(num_images / grid_cols))

    plt.figure(figsize=(grid_cols * 3, grid_rows * 3))
    plt.suptitle(title)
    batch = batch[0]
    labels = labels[0]
    for idx in range(num_images):
        img = batch[idx]  # Shape: (1, height, width)
        label = labels[idx]
        
        # Remove the singleton dimension
        img = img[0]  # Now img has shape (height, width)

        # Ensure the image is in uint8 format
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # Debug: Print image details
        print(f"Image {idx} - shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")

        # Display the image
        plt.subplot(grid_rows, grid_cols, idx + 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.title(f'Label: {label}')

    plt.tight_layout()
    plt.show()
