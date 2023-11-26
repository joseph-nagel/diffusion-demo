'''DDPM training on swiss roll data.'''

import argparse

import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    StochasticWeightAveraging
)

from diffusion import (
    make_swissroll_2d,
    DDPMTab
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt-file', type=str, required=False, help='checkpoint for resuming')

    parser.add_argument('--save-dir', type=str, default='run/', help='save dir')
    parser.add_argument('--name', type=str, default='swissroll', help='experiment name')
    parser.add_argument('--version', type=str, required=False, help='experiment version')

    parser.add_argument('--num-samples', type=int, default=2000, help='number of data samples')
    parser.add_argument('--noise-level', type=float, default=0.5, help='noise level')
    parser.add_argument('--scaling', type=float, default=0.1, help='scaling parameter')
    parser.add_argument('--val-size', type=float, default=0.2, help='validation set size')

    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')

    parser.add_argument('--mid-features', type=int, nargs='+', default=[128, 128, 128], help='feature numbers')
    parser.add_argument('--activation', type=str, default='leaky_relu', help='nonlinearity type')
    parser.add_argument('--embed-dim', type=int, default=128, help='embedding dimension')

    parser.add_argument('--beta-mode', type=str, default='cosine', help='beta schedule mode')
    parser.add_argument('--beta-range', type=int, nargs='+', default=[1e-04, 0.02], help='beta range')
    parser.add_argument('--cosine-s', type=float, default=0.008, help='offset for cosine schedule')
    parser.add_argument('--sigmoid-range', type=int, nargs='+', default=[-5, 5], help='sigmoid range')
    parser.add_argument('--num-steps', type=int, default=500, help='number of time steps')

    parser.add_argument('--criterion', type=str, default='mse', help='loss function criterion')
    parser.add_argument('--lr', type=float, default=1e-03, help='optimizer learning rate')

    parser.add_argument('--max-epochs', type=int, default=1000, help='max. number of training epochs')
    parser.add_argument('--gradient-clip-val', type=float, default=0.5, help='gradient clipping value')
    parser.add_argument('--gradient-clip-algorithm', type=str, default='norm', help='gradient clipping mode')

    parser.add_argument('--swa-lrs', type=float, default=1e-04, help='SWA learning rate')
    parser.add_argument('--swa-epoch-start', type=float, default=0.8, help='SWA start epoch')
    parser.add_argument('--annealing-epochs', type=int, default=10, help='SWA annealing epochs')
    parser.add_argument('--annealing-strategy', type=str, default='cos', help='SWA annealing strategy')

    args = parser.parse_args()

    return args


def main(args):

    # create datasets
    x_train, x_val = make_swissroll_2d(
        num_samples=args.num_samples,
        noise_level=args.noise_level,
        scaling=args.scaling,
        val_size=args.val_size,
        as_tensor=True
    )

    train_set = TensorDataset(x_train)
    val_set = TensorDataset(x_val)

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
    ddpm = DDPMTab(
        in_features=2,
        mid_features=args.mid_features,
        activation=args.activation,
        embed_dim=args.embed_dim,
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

    ckpt_callback = ModelCheckpoint(
        filename='best',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True
    )
    callbacks_list = [ckpt_callback]

    if args.swa_lrs > 0:
        swa_callback = StochasticWeightAveraging(
            swa_lrs=args.swa_lrs,
            swa_epoch_start=args.swa_epoch_start,
            annealing_epochs=args.annealing_epochs,
            annealing_strategy=args.annealing_strategy
        )
        callbacks_list.append(swa_callback)

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    gradient_clip_val = args.gradient_clip_val if args.gradient_clip_val > 0 else None

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks_list,
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

