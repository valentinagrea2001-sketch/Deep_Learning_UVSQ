# Deep Learning exercises at UVSQ and Paris-Saclay

Here is the content of the course of Deep Learning at the University of Versailles Saint-Quentin, belonging to the one-year master's degree M2 of Paris-Saclay in "Mathematics and Statistical Learning":

0) Review of Python classes and subclasses

1) Introduction to Feedforward Neural Networks == MLPs
   
2) Activation functions plots, with pros and cons

3) Logistic regression: a multiclassifier with 1 hidden layer

   > A comparison between: bootstrap random forest method, Sklearn MLP classifier, and PyTorch classifier

4) - MLP multiclassifier with more hidden layers

   - Comparison between optimization methods Gradient Descent, Stochastic Gradient Descent, Adam

   > Adam provides the best results in terms of accuracy

5) An example of MLP trained on a OpenML dataset

6) Datasets, Dataloads, Subsets

7) A NN with more hidden layers and a comparison between optimization methods (GD, SGD, mini-batch SGD, Adam)

   > Here implemented mini-batch SGD

8) Convolutional Neural Networks

------------

*For each topic, you will find an exercise sheet in Python.*

------------

## Employed libraries:

numpy, scikit-learn, torch.nn, torch.optim, torchinfo, collections.abc, matplotlib.pyplot, openml, pathlib

## Employed datasets:

MNIST from torch.datasets
