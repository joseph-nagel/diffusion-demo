'''DDPM training on MNIST.'''

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from diffusion import DDPM2d


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt-file', type=str, required=False, help='checkpoint for resuming')

    parser.add_argument('--data-dir', type=str, default='data', help='data dir')

    parser.add_argument('--save-dir', type=str, default='.', help='save dir')
    parser.add_argument('--name', type=str, default='mnist', help='experiment name')
    parser.add_argument('--version', type=str, required=False, help='experiment version')

    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')

    parser.add_argument('--mid-channels', type=int, nargs='+', default=[16, 32, 64], help='channel numbers')
    parser.add_argument('--kernel-size', type=int, default=3, help='conv. kernel size')
    parser.add_argument('--padding', type=int, default=1, help='padding parameter')
    parser.add_argument('--norm', type=str, default='batch', help='normalization type')
    parser.add_argument('--activation', type=str, default='leaky_relu', help='nonlinearity type')
    parser.add_argument('--embed-dim', type=int, default=128, help='embedding dimension')
    parser.add_argument('--num-resblocks', type=int, default=3, help='number of residual blocks')
    parser.add_argument('--upsample-mode', type=str, default='conv_transpose', help='conv. upsampling mode')

    parser.add_argument('--beta-mode', type=str, default='cosine', help='beta schedule mode')
    parser.add_argument('--beta-range', type=int, nargs='+', default=[1e-04, 0.02], help='beta range')
    parser.add_argument('--cosine-s', type=float, default=0.008, help='offset for cosine schedule')
    parser.add_argument('--sigmoid-range', type=int, nargs='+', default=[-5, 5], help='sigmoid range')
    parser.add_argument('--num-steps', type=int, default=1000, help='number of time steps')

    parser.add_argument('--criterion', type=str, default='mse', help='loss function criterion')
    parser.add_argument('--lr', type=float, default=1e-04, help='optimizer learning rate')

    parser.add_argument('--max-epochs', type=int, default=1000, help='max. number of training epochs')
    parser.add_argument('--gradient-clip-val', type=float, default=0.2, help='gradient clipping value')
    parser.add_argument('--gradient-clip-algorithm', type=str, default='norm', help='gradient clipping mode')

    args = parser.parse_args()
    return args


def main(args):

    # create datasets
    transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.ToTensor()
    ]) # TODO: refine data augmentation

    train_set = datasets.MNIST(
        args.data_dir,
        train=True,
        transform=transform,
        download=True
    )

    val_set = datasets.MNIST(
        args.data_dir,
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    # create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # initialize model
    ddpm = DDPM2d(
        in_channels=1,
        mid_channels=args.mid_channels,
        kernel_size=args.kernel_size,
        padding=args.padding,
        norm=args.norm,
        activation=args.activation,
        embed_dim=args.embed_dim,
        num_resblocks=args.num_resblocks,
        upsample_mode=args.upsample_mode,
        beta_mode=args.beta_mode,
        beta_range=args.beta_range,
        cosine_s=args.cosine_s,
        sigmoid_range=args.sigmoid_range,
        num_steps=args.num_steps,
        criterion=args.criterion,
        lr=args.lr
    )

    # create trainer
    logger = TensorBoardLogger(args.save_dir, name=args.name, version=args.version)
    # logger.log_hyperparams(vars(args)) # save all (hyper)params

    checkpoint_callback = ModelCheckpoint(
        filename='best',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True
    )

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    gradient_clip_val = None if args.gradient_clip_val <= 0 else args.gradient_clip_val

    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator=accelerator,
        devices=1,
        max_epochs=args.max_epochs,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        log_every_n_steps=len(train_loader),
        enable_progress_bar=True
    )

    # check validation loss
    trainer.validate(
        ddpm,
        dataloaders=val_loader,
        verbose=False,
        ckpt_path=args.ckpt_file
    )

    # train model
    trainer.fit(
        ddpm,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.ckpt_file
    )


if __name__ == '__main__':

    args = parse_args()
    main(args)

