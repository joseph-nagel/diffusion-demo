'''DDPM training on MNIST.'''

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from diffusion import (
    DDPM,
    UNet,
    make_beta_schedule
)


def main(args):

    # create datasets
    transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.ToTensor()
    ])

    train_set = datasets.MNIST(args.data_dir,
                               train=True,
                               transform=transform,
                               download=True)

    val_set = datasets.MNIST(args.data_dir,
                             train=False,
                             transform=transforms.ToTensor(),
                             download=True)


    # create data loaders
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              drop_last=True,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)

    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            drop_last=False,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True)


    # initialize model
    eps_model = UNet.from_params(in_channels=1,
                                 mid_channels=args.mid_channels,
                                 kernel_size=args.kernel_size,
                                 padding=args.padding,
                                 norm=args.norm,
                                 activation=args.activation,
                                 embed_dim=args.embed_dim,
                                 num_resblocks=args.num_resblocks,
                                 upsample_mode=args.upsample_mode)

    if args.beta_mode == 'quadratic':
        beta_opts = {'beta_range': args.beta_range}
    elif args.beta_mode == 'cosine':
        beta_opts = {'cosine_s': args.cosine_s}
    elif args.beta_mode == 'sigmoid':
        beta_opts = {'sigmoid_range': args.sigmoid_range}

    betas = make_beta_schedule(num_steps=args.num_steps, mode=args.beta_mode, **beta_opts)

    ddpm = DDPM(eps_model=eps_model, betas=betas, criterion='mse')


    # train model
    logger = TensorBoardLogger(args.save_dir, name=args.name, version=args.version)
    logger.log_hyperparams(vars(args)) # save all (hyper)params

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', save_last=True)

    trainer = Trainer(logger=logger,
                      callbacks=[checkpoint_callback],
                      accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      devices=1,
                      max_epochs=args.max_epochs,
                      log_every_n_steps=len(train_loader),
                      enable_progress_bar=True)

    trainer.validate(ddpm, dataloaders=val_loader, verbose=False) # check validation loss before training
    trainer.fit(ddpm, train_dataloaders=train_loader, val_dataloaders=val_loader) # start training


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data', help='data dir')

    parser.add_argument('--save-dir', type=str, default='.', help='save dir')
    parser.add_argument('--name', type=str, default='lightning_logs', help='experiment name')
    parser.add_argument('--version', type=str, required=False, help='experiment version')

    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')

    parser.add_argument('--mid-channels', type=int, nargs='+', default=[8, 16, 32], help='channel numbers')
    parser.add_argument('--kernel-size', type=int, default=3, help='kernel size')
    parser.add_argument('--padding', type=int, default=1, help='padding parameter')
    parser.add_argument('--norm', type=str, default='batch', help='normalization type')
    parser.add_argument('--activation', type=str, default='relu', help='nonlinearity type')
    parser.add_argument('--embed-dim', type=int, default=100, help='embedding dimension')
    parser.add_argument('--num-resblocks', type=int, default=3, help='number of residual blocks')
    parser.add_argument('--upsample-mode', type=str, default='bilinear_conv', help='conv. upsampling mode')

    parser.add_argument('--beta-mode', type=str, default='cosine', help='beta schedule mode')
    parser.add_argument('--beta-range', type=int, nargs='+', default=[1e-04, 0.02], help='beta range')
    parser.add_argument('--cosine-s', type=float, default=0.008, help='offset for cosine schedule')
    parser.add_argument('--sigmoid-range', type=int, nargs='+', default=[-5, 5], help='sigmoid range')
    parser.add_argument('--num-steps', type=int, default=1000, help='number of noise process steps')

    parser.add_argument('--max-epochs', type=int, default=2000, help='max. number of training epochs')
    args = parser.parse_args()

    main(args)

