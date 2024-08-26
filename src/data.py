import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def load_pretrain_data(input_path, scaler_path, group_cols=None):
    input_data = pd.read_csv(input_path)

    cont_cols = input_data.filter(regex="X_*").columns
    scaler = StandardScaler()
    input_data[cont_cols] = scaler.fit_transform(input_data[cont_cols])

    scaler_folder = os.path.dirname(scaler_path)
    if not os.path.exists(scaler_folder):
        os.makedirs(scaler_folder)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    if group_cols is None:
        return input_data[cont_cols].values

    input_groups = []
    for _, group_data in input_data.groupby(group_cols):
        input_groups.append(group_data[cont_cols].values)
    return input_groups


def generate_mask(input_raw, mask_prob):
    return np.random.binomial(1, mask_prob, input_raw.shape)


def generate_pretext(input_raw, input_mask):
    input_bar = np.zeros(input_raw.shape)
    for i in range(input_raw.shape[1]):
        indices = np.random.permutation(input_raw.shape[0])
    input_bar[:, i] = input_raw[indices, i]

    input_tilde = input_raw * (1 - input_mask) + input_bar * input_mask
    input_mask = 1 * (np.abs(input_raw - input_tilde) > 1e-6)
    return input_tilde, input_mask


def generate_pretraining_data(input_path, scaler_path, group_cols, mask_prob):
    if group_cols is None:
        input_unlab = load_pretrain_data(input_path, scaler_path, group_cols)
        input_mask = generate_mask(input_unlab, mask_prob)
        input_tilde, input_mask = generate_pretext(input_unlab, input_mask)
        return input_tilde, input_unlab, input_mask
    else:
        input_groups = load_pretrain_data(input_path, scaler_path, group_cols)

    input_tilde, input_mask = [], []
    for input_unlab_part in tqdm(input_groups, desc="group processsing"):
        input_mask_part = generate_mask(input_unlab_part, mask_prob)
        input_tilde_part, input_mask_part = generate_pretext(input_unlab_part, input_mask_part)
        input_tilde.append(input_tilde_part)
        input_mask.append(input_mask_part)
    input_unlab = np.concatenate(input_groups, axis=0)
    input_tilde = np.concatenate(input_tilde, axis=0)
    input_mask = np.concatenate(input_mask, axis=0)

    return input_unlab, input_tilde, input_mask


def generate_test_data(input_path, scaler_path):
    input_data = pd.read_csv(input_path)

    cont_cols = input_data.filter(regex="X_*").columns
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    input_data[cont_cols] = scaler.transform(input_data[cont_cols])

    input_unlab = input_data[cont_cols].values
    input_tilde = input_unlab.copy()
    input_mask = np.zeros(input_unlab.shape)

    return input_unlab, input_tilde, input_mask


class PretrainDataset(Dataset):
    def __init__(self, input_path, scaler_path, group_cols, mask_prob, mode="train"):
        if mode == "train":
            input_unlab, input_tilde, input_mask = generate_pretraining_data(
                input_path, scaler_path, group_cols, mask_prob)
        elif mode == "test":
            input_unlab, input_tilde, input_mask = generate_test_data(input_path, scaler_path)

        self.input_unlab = input_unlab.astype(np.float32)
        self.input_tilde = input_tilde.astype(np.float32)
        self.input_mask = input_mask.astype(np.float32)

    def __getitem__(self, index):
        return self.input_unlab[index], self.input_tilde[index], self.input_mask[index]

    def __len__(self):
        return len(self.input_unlab)
