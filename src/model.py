import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import PretrainDataset


class PretrainModel(nn.Module):
    def __init__(self, num_feat, num_hidden):
        super(PretrainModel, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(num_feat, num_hidden), nn.LeakyReLU())
        self.mask_decoder = nn.Sequential(nn.Linear(num_hidden, num_feat), nn.Sigmoid())
        self.feat_decoder = nn.Linear(num_hidden, num_feat)

    def forward(self, inputs):
        hidden = self.encoder(inputs)
        return self.mask_decoder(hidden), self.feat_decoder(hidden)


class LitPretrainModel(pl.LightningModule):
    def __init__(self, running_config):
        super().__init__()
        self.save_hyperparameters()

        self.data_config = running_config["data_config"]
        self.model_config = running_config["model_config"]

        self.model = PretrainModel(self.model_config["num_feat"], self.model_config["num_hidden"])
        self.mask_criterion = nn.BCELoss()
        self.feat_criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        input_unlab, input_tilde, input_mask = batch
        output_mask, output_feat = self.model(input_tilde)

        mask_loss = self.mask_criterion(output_mask, input_mask)
        feat_loss = self.feat_criterion(output_feat, input_unlab) * self.model_config["alpha"]
        total_loss = mask_loss + feat_loss

        self.log("feat_loss", feat_loss, prog_bar=True)
        self.log("mask_loss", mask_loss, prog_bar=True)
        self.log("total_loss", total_loss)
        return total_loss

    def train_dataloader(self):
        train_set = PretrainDataset(**self.data_config)
        train_loader = DataLoader(train_set, shuffle=True, batch_size=self.model_config["batch_size"])
        return train_loader

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input_unlab, _, _ = batch
        return self.model.encoder(input_unlab)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.model_config["lr"])
