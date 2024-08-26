import argparse
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from torch.utils.data import DataLoader

from src.data import PretrainDataset
from src.model import LitPretrainModel


def run(args):
    model = LitPretrainModel.load_from_checkpoint(args.model_path)
    trainer = pl.Trainer(devices=1, default_root_dir="outputs")

    train_data = pd.read_csv(args.train_path)
    train_set = PretrainDataset(input_path=args.train_path, scaler_path=args.scaler_path,
                                group_cols=None, mask_prob=0.0, mode="test")
    train_loader = DataLoader(train_set, shuffle=False, batch_size=args.batch_size, num_workers=4)

    embedding = trainer.predict(model, train_loader)
    X_train = torch.cat(embedding, dim=0).detach().cpu().numpy()
    y_train = train_data["Y"].values
    print("X_Train Shape:", X_train.shape, "y_train Shape:", y_train.shape)

    test_data = pd.read_csv(args.test_path)
    test_set = PretrainDataset(input_path=args.test_path, scaler_path=args.scaler_path,
                               group_cols=None, mask_prob=0.0, mode="test")
    test_loader = DataLoader(test_set, shuffle=False, batch_size=args.batch_size, num_workers=4)

    embedding = trainer.predict(model, test_loader)
    X_test = torch.cat(embedding, dim=0).detach().cpu().numpy()
    y_test = test_data["Y"].values
    print("X_Test Shape:", X_test.shape, "y_test Shape:", y_test.shape)

    knn = KNeighborsRegressor(n_neighbors=args.n_neighbors, weights="distance")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    metric_dict = {
        "R2": r2_score(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
        "PEARSON": pearsonr(y_pred, y_test)[0]
    }
    print(metric_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, required=True, help="train file path")
    parser.add_argument("--test_path", type=str, required=True, help="test file path")
    parser.add_argument("--scaler_path", type=str, required=True, help="standard scaler save path")
    parser.add_argument("--model_path", type=str, required=True, help="model save path")

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_neighbors", type=int, default=5)
    args = parser.parse_args()

    run(args)
