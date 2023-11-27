# PyTorch denoising diffusion demo

The repository contains a simple PyTorch-based demonstration of denoising diffusion models.
It just aims at providing a basic understanding of this generative modeling approach.

A short theoretical intro to the topic can be found [here](notebooks/ddpm_intro.ipynb).
Two example applications establish a small experimentation playground.
They are prepared in such a way that they can be easily modified.

## Swiss roll

First, [the script](./train_ddpm_swissroll.py) trains a generative DDPM model on a 2D Swiss roll distribution.
It exposes a number of arguments that allow one to adjust the problem setup and model definition.
The training script can be run with reasonable default settings by:
```
python train_ddpm_swissroll.py
```
After the training has finished, the final model can be tested and analyzed in [this notebook](notebooks/ddpm_swissroll.ipynb).

For monitoring the experiment, one can locally run a TensorBoard server by `tensorboard --logdir run/swissroll/`.
It can be reached under [localhost:6006](http://localhost:6006) per default in your browser.
As an alternative, one may use MLfLow for managing experiments.
In this case, one can launch the training as `python train_ddpm_swissroll.py --logger mlflow`
and set up a tracking server by `mlflow server --backend-store-uri file:./run/mlruns/`.
It can then be reached under [localhost:5000](http://localhost:5000).

## MNIST

The second application is based on the MNIST dataset.
A DDPM generating images of handwritten digits can be learned by running [the script](./train_ddpm_mnist.py).
This is done by calling:
```
python train_ddpm_mnist.py
```
[A dedicated notebook](notebooks/ddpm_mnist.ipynb) is provided in order to test the trained model.

