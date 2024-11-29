# CNN Framework

This repository contains my own implementation of a Convolutional Neural Network (CNN) framework for model training and creation.

### Highlights
- **Optimized Computations:** Used dot products where applicable to enhance computational speed.
- **Tested with LeNet-5:** Successfully implemented the LeNet-5 architecture to validate the framework. Tested on MNIST dataset with preprocessing (96.5% Accuracy). Minor adjustments may be required to implement other CNN architectures.

### Results
![Result Image](https://private-user-images.githubusercontent.com/109877609/391038487-acfaaa7f-0647-44d1-940e-d51309137ff9.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzI4NzM0MzQsIm5iZiI6MTczMjg3MzEzNCwicGF0aCI6Ii8xMDk4Nzc2MDkvMzkxMDM4NDg3LWFjZmFhYTdmLTA2NDctNDRkMS05NDBlLWQ1MTMwOTEzN2ZmOS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMTI5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTEyOVQwOTM4NTRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0xZGVjNzkzZDViNmIzM2NhOTU3ODRmOTgzYTQ4NWY3Yjk5MWIxODRhZjkyYWEwOTQ5ODc2ZTllYTIzOWI2ZGQ1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.rrBnGVkgNVAP_kuGdWyAPIT5Whe0Dr3nBj1mdzR38gc)

### Download
[Link to download the trained test model](To add)

### Notes
- **Partial Connections in Layer C3:** I did not use partial connections for layer C3, as described in the original LeNet-5 paper, to simplify and speed up computations by avoiding nested loops.
- **Classification Layer:** The final layer uses Softmax for classification instead of the Radial Basis Function (RBF) described in the original paper.

### Reference
LeCun et al.'s original paper on LeNet-5:  
[Gradient-Based Learning Applied to Document Recognition (1998)](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
