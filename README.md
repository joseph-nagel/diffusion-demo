# PyTorch denoising diffusion demo

The repository contains a simple PyTorch-based demonstration of denoising diffusion models.
It just aims at providing a basic understanding of this generative modeling approach.

## Overview
A short theoretical intro to the topic can be found [here](notebooks/ddpm_intro.ipynb).
Two example applications are provided as a small experimentation playground.
First, [this notebook](notebooks/ddpm_swissroll.ipynb) considers the Swiss roll distribution.
Second, a DDPM for the MNIST dataset can be learned by running `python train_ddpm_mnist.py`.
The trained model may then be analyzed in [another notebook](notebooks/ddpm_mnist.ipynb).

