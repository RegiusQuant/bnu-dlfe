import argparse

import pytorch_lightning as pl

from src.model import LitPretrainModel


def run(args):
    running_config = generate_running_config(args)

    model = LitPretrainModel(running_config)
    trainer = pl.Trainer(max_epochs=args.max_epochs, devices=1, default_root_dir="outputs",
                         reload_dataloaders_every_n_epochs=1)
    trainer.fit(model)
    trainer.save_checkpoint(args.model_path)


def generate_running_config(args):
    running_config = {
        "data_config": {
            "input_path": args.input_path,
            "scaler_path": args.scaler_path,
            "group_cols": args.group_cols,
            "mask_prob": args.mask_prob,
        },
        "model_config": {
            "num_feat": args.num_feat,
            "num_hidden": args.num_hidden,
            "alpha": args.alpha,
            "batch_size": args.batch_size,
            "lr": args.lr,
        }
    }
    return running_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, required=True, help="input pretraining file path")
    parser.add_argument("--scaler_path", type=str, required=True, help="standard scaler save path")
    parser.add_argument("--group_cols", nargs="+", help="which columns to group data")
    parser.add_argument("--mask_prob", type=float, default=0.2, help="mask probability in pretext")

    parser.add_argument("--num_feat", type=int, required=True, help="number of input features (X_*)")
    parser.add_argument("--num_hidden", type=int, default=64, help="number of hidden features")
    parser.add_argument("--alpha", type=float, default=0.5, help="loss coefficient")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--model_path", type=str, default="outputs/model.ckpt", help="model save path")

    args = parser.parse_args()

    pl.seed_everything(42)
    run(args)
