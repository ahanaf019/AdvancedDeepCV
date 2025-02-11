# Advanced Deep Computer Vision

In this repository, I explore various deep learning-based computer vision models and methods, covering **supervised, unsupervised, and semi-supervised** approaches. The goal is to assess the effectiveness of these methods using the **STL-10 dataset**, with at most **100 images per class** for evaluation.

## Methods Covered (will add progressively)

- **[Supervised Learning](supervised/ReadMe.md)**
- **Unsupervised Learning**
- **Semi-Supervised Learning**

## Dataset: STL-10

STL-10 is an image recognition dataset inspired by CIFAR-10 dataset with some improvements. With a corpus of 100,000 unlabeled images and 500 training images, this dataset is best for developing unsupervised feature learning, deep learning, self-taught learning algorithms. Unlike CIFAR-10, the dataset has a higher resolution which makes it a challenging benchmark for developing more scalable unsupervised learning methods.

- 10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
- Images are 96x96 pixels, color
- 500 training images (10 pre-defined folds), 800 test images per class
- 100,000 unlabeled images for unsupervised learning. These examples are extracted from a similar but broader distribution of images. For instance, it contains other types of animals (bears, rabbits, etc.) and vehicles (trains, buses, etc.) in addition to the ones in the labeled set
- Images were acquired from labeled examples on ImageNet

The **STL-10 dataset** is used with a constraint of **100 images per class** to evaluate the robustness and generalization ability of each method. STL-10 contains **10 classes** of images, originally designed for **semi-supervised learning**.
