'''
Training script.

Example
-------
python scripts/main.py fit --config config/swissroll.yaml

'''

from lightning.pytorch.cli import LightningCLI


def main():
    cli = LightningCLI()


if __name__ == '__main__':
    main()
