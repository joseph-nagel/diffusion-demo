'''DDPM training on Swiss roll data.'''

from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
    StochasticWeightAveraging
)

from diffusion import make_swiss_roll_2d, DDPMTab


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--random-seed', type=int, default=12345, help='Random seed')

    parser.add_argument('--ckpt-file', type=Path, required=False, help='Checkpoint for resuming')

    parser.add_argument('--logger', type=str, default='tensorboard', help='Logger')
    parser.add_argument('--save-dir', type=Path, default='run/', help='Save dir')
    parser.add_argument('--name', type=str, default='swissroll', help='Experiment name')
    parser.add_argument('--version', type=str, required=False, help='Experiment version')

    parser.add_argument('--log-every-n-steps', type=int, default=50, help='How often to log train steps')
    parser.add_argument('--check-val-every-n-epoch', type=int, default=1, help='How many epochs between val. checks')

    parser.add_argument('--save-top-k', type=int, default=1, help='Number of best models to save')
    parser.add_argument('--save-every-n-epochs', type=int, default=1, help='Regular checkpointing interval')

    parser.add_argument('--num-samples', type=int, default=3000, help='Number of data samples')
    parser.add_argument('--noise-level', type=float, default=0.5, help='Noise level')
    parser.add_argument('--scaling', type=float, default=0.15, help='Scaling parameter')
    parser.add_argument('--val-size', type=float, default=0.2, help='Validation set size')

    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers')

    parser.add_argument('--mid-features', type=int, nargs='+', default=[128, 128, 128], help='Feature numbers')
    parser.add_argument('--activation', type=str, default='leaky_relu', help='Nonlinearity type')
    parser.add_argument('--embed-dim', type=int, default=128, help='Dimension of the time embedding')

    parser.add_argument('--num-steps', type=int, default=500, help='Number of time steps')
    parser.add_argument('--schedule', type=str, default='cosine', help='Noise schedule mode')
    parser.add_argument('--beta-range', type=float, nargs='+', default=[1e-04, 0.02], help='Beta range')
    parser.add_argument('--cosine-s', type=float, default=0.008, help='Offset for cosine schedule')
    parser.add_argument('--sigmoid-range', type=float, nargs='+', default=[-5.0, 5.0], help='Sigmoid range')

    parser.add_argument('--criterion', type=str, default='mse', help='Loss function criterion')

    parser.add_argument('--lr', type=float, default=1e-03, help='Initial learning rate')
    parser.add_argument('--lr-schedule', type=str, default='constant', choices=['constant', 'cosine'], help='LR schedule type')
    parser.add_argument('--lr-interval', type=str, default='epoch', choices=['epoch', 'step'], help='LR update interval')
    parser.add_argument('--lr-warmup', type=int, default=0, help='Warmup steps/epochs')

    parser.add_argument('--max-epochs', type=int, default=1000, help='Max. number of training epochs')

    parser.add_argument('--patience', type=int, default=0, help='Early stopping patience')

    parser.add_argument('--swa-lrs', type=float, default=1e-04, help='SWA learning rate')
    parser.add_argument('--swa-epoch-start', type=float, default=0.7, help='SWA start epoch')
    parser.add_argument('--annealing-epochs', type=int, default=10, help='SWA annealing epochs')
    parser.add_argument('--annealing-strategy', type=str, default='cos', help='SWA annealing strategy')

    parser.add_argument('--gradient-clip-val', type=float, default=0.0, help='Gradient clipping value')
    parser.add_argument('--gradient-clip-algorithm', type=str, default='norm', help='Gradient clipping mode')

    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--cpu', dest='gpu', action='store_false', help='Do not use GPU')
    parser.set_defaults(gpu=False)

    args = parser.parse_args()

    return args


def main(args):

    # set random seeds
    if args.random_seed is not None:
        _ = seed_everything(
            args.random_seed,
            workers=args.num_workers > 0
        )

    # create datasets
    x_train, x_val = make_swiss_roll_2d(
        num_samples=args.num_samples,
        noise_level=args.noise_level,
        scaling=args.scaling,
        random_state=None,
        test_size=args.val_size
    )

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)

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
        num_steps=args.num_steps,
        schedule=args.schedule,
        beta_range=args.beta_range,
        cosine_s=args.cosine_s,
        sigmoid_range=args.sigmoid_range,
        criterion=args.criterion,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        lr_interval=args.lr_interval,
        lr_warmup=args.lr_warmup
    )

    # set accelerator
    if args.gpu:
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    else:
        accelerator = 'cpu'

    # create logger
    if args.logger == 'tensorboard':
        logger = TensorBoardLogger(
            save_dir=args.save_dir,
            name=args.name,
            version=args.version
        )
    elif args.logger == 'mlflow':
        logger = MLFlowLogger(
            experiment_name=args.name,
            run_name=args.version,
            save_dir=args.save_dir / 'mlruns',
            log_model=True
        )
    else:
        raise ValueError('Unknown logger: {}'.format(args.logger))

    # set up LR monitor
    lr_monitor = LearningRateMonitor(logging_interval=None)

    callbacks = [lr_monitor]

    # set up checkpointing
    save_top_ckpt = ModelCheckpoint(
        filename='best',
        monitor='val_loss',
        mode='min',
        save_top_k=args.save_top_k
    )

    save_every_ckpt = ModelCheckpoint(
        filename='{epoch}',
        save_top_k=-1,
        every_n_epochs=args.save_every_n_epochs,
        save_last=True
    )

    callbacks.extend([save_top_ckpt, save_every_ckpt])

    # set up early stopping
    if args.patience > 0:
        early_stopping = EarlyStopping('val_loss', patience=args.patience)
        callbacks.append(early_stopping)

    # set up weight averaging
    if args.swa_lrs > 0:
        swa = StochasticWeightAveraging(
            swa_lrs=args.swa_lrs,
            swa_epoch_start=args.swa_epoch_start,
            annealing_epochs=args.annealing_epochs,
            annealing_strategy=args.annealing_strategy
        )
        callbacks.append(swa)

    # set up gradient clipping
    if args.gradient_clip_val > 0:
        gradient_clip_val = args.gradient_clip_val
        gradient_clip_algorithm = args.gradient_clip_algorithm
    else:
        gradient_clip_val = None
        gradient_clip_algorithm = None

    # initialize trainer
    trainer = Trainer(
        accelerator=accelerator,
        logger=logger,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        deterministic=args.random_seed is not None
    )

    # check validation loss
    trainer.validate(
        model=ddpm,
        dataloaders=val_loader,
        verbose=False,
        ckpt_path=args.ckpt_file
    )

    # train model
    trainer.fit(
        model=ddpm,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.ckpt_file
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)
