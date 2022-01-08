import argparse
import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.data import PretrainDataset
from src.model import LitPretrainModel


def run(args):
    save_path = args.save_path
    save_folder, _ = os.path.split(save_path)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    test_test = PretrainDataset(input_path=args.input_path, scaler_path=args.scaler_path,
                                group_cols=None, mask_prob=0.0, mode="test")
    test_loader = DataLoader(test_test, shuffle=False, batch_size=args.batch_size)
    model = LitPretrainModel.load_from_checkpoint(args.model_path)
    trainer = pl.Trainer(gpus=1, default_root_dir="outputs")

    embedding = trainer.predict(model, test_loader)
    embedding = torch.cat(embedding, dim=0).detach().cpu().numpy()
    np.save(args.save_path, embedding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, required=True, help="input pretraining file path")
    parser.add_argument("--scaler_path", type=str, required=True, help="standard scaler save path")
    parser.add_argument("--model_path", type=str, required=True, help="model save path")
    parser.add_argument("--save_path", type=str, required=True, help="feature save path")

    parser.add_argument("--batch_size", type=int, default=512)

    args = parser.parse_args()

    run(args)
