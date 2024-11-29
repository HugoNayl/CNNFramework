# CNN Framework

This repository contains my own implementation of a Convolutional Neural Network (CNN) framework for model training and creation.

### Highlights
- **Optimized Computations:** Used dot products where applicable to enhance computational speed.
- **Tested with LeNet-5:** Successfully implemented the LeNet-5 architecture to validate the framework. Minor adjustments may be required to implement other CNN architectures.

### Download
[Link to download the trained test model](To add)

### Notes
- **Partial Connections in Layer C3:** I did not use partial connections for layer C3, as described in the original LeNet-5 paper, to simplify and speed up computations by avoiding nested loops.
- **Classification Layer:** The final layer uses Softmax for classification instead of the Radial Basis Function (RBF) described in the original paper.

### Reference
LeCun et al.'s original paper on LeNet-5:  
[Gradient-Based Learning Applied to Document Recognition (1998)](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
