# Deep Learning exercises at UVSQ and Paris-Saclay

Here is the content of the course of Deep Learning at the University of Versailles Saint-Quentin, belonging to the one-year master's degree M2 of Paris-Saclay in "Mathematics and Statistical Learning":

0) Review of Python classes and subclasses.

1) Introduction to Feedforward Neural Networks == MLPs.
   
2) Activation functions plots, with pros and cons.

3) Logistic regression: a multiclassifier with 1 hidden layer.

   > A comparison between: bootstrap random forest method, Sklearn MLP classifier, and PyTorch classifier.

4) - MLP multiclassifier with more hidden layers.

   - Comparison between optimization methods Gradient Descent, Stochastic Gradient Descent, Adam.

   > Adam provides the best results in terms of accuracy.

5) An example of MLP trained on a OpenML dataset.

6) Datasets, Subsets, Dataloads.

   > The 'in practice' approach to prepare a dataset for the training and evaluation of a MLP.

7) Mini-batch training.

   > THIS FILE CONTAINS THE BEST PRACTICE UNTIL HERE!!

   1) Preparing the dataset with Dataset, Subsets, DataLoader
   2) Creating the nn.Module class of the MLP classifier
   3) Training and evaluating the model with batching
   4) One Hot Encoding training (if labels are categorical)

8) A NN with more hidden layers and a comparison between optimization methods (GD, SGD, mini-batch SGD, Adam).

   > Here implemented mini-batch SGD

9) Convolutional Neural Networks.

10) Natural Language processing and Transformers.

------------

*For each topic, you will find an exercise sheet in Python.*

------------

## Employed libraries:

numpy, scikit-learn, torch.nn, torch.optim, torchinfo, collections.abc, matplotlib.pyplot, openml, pathlib

## Employed datasets:

MNIST from torch.datasets
