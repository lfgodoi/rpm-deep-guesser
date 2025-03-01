#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Dual Spectrum Net (DSN)

The Dual Spectrum Net architecture for predicting RPM value

"""

# Importing packages and modules
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pickle
import json
import datetime
from nn_utils import EarlyStopping
from config import Config

# Dataset admitted by the Dual Spectrum Net
class DualSpectrumNetDataset(Dataset):

    # Constructor
    def __init__(self, original_features, envelope_features, labels):

        # Instance parameters
        self.original_features = torch.from_numpy(original_features).float()
        self.envelope_features = torch.from_numpy(envelope_features).float()
        self.labels = torch.from_numpy(labels).float()

        # Configuration parameters
        config = Config()
        self.model_path = config.model_path

        # Class parameters
        self.signature = "dual_spectrum_net"

    # Total number of features
    def __len__(self):
        return len(self.labels)

    # Return by index
    def __getitem__(self, idx):
        return (self.original_features[idx], self.envelope_features[idx]), self.labels[idx]

# Dual Spectrum Net
class DualSpectrumNet(nn.Module):

    # Constructor
    def __init__(self, num_features, neurons):
        super().__init__()

        # Instance parameters
        self.num_features = num_features
        self.neurons = neurons

        # Original Spectrum Subnet
        self.original_spectrum_subnet = nn.Sequential(
            nn.Linear(num_features, neurons[0]),
            nn.ReLU(),
            nn.Linear(neurons[0], neurons[1]),
            nn.ReLU(),
            nn.Linear(neurons[1], neurons[2]),
            nn.ReLU()
        )

        # Envelope Spectrum Subnet
        self.envelope_spectrum_subnet = nn.Sequential(
            nn.Linear(num_features, neurons[0]),
            nn.ReLU(),
            nn.Linear(neurons[0], neurons[1]),
            nn.ReLU(),
            nn.Linear(neurons[1], neurons[2]),
            nn.ReLU()
        )

        # Predictor Subnet
        self.predictor_subnet = nn.Sequential(
            nn.Linear(int(2*neurons[2]), 1)
        )

        # Initializing the trainable parameters
        def initialize_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
        self.original_spectrum_subnet.apply(initialize_weights)
        self.envelope_spectrum_subnet.apply(initialize_weights)
        self.predictor_subnet.apply(initialize_weights)

    # Forward propagation
    def forward(self, original_features, envelope_features):
        original_encoding = self.original_spectrum_subnet(original_features)
        envelope_encoding = self.envelope_spectrum_subnet(envelope_features)
        combined_encoding = torch.cat((original_encoding, envelope_encoding), dim=1)
        predictions = self.predictor_subnet(combined_encoding)
        return predictions

    # Training
    def fit(self,
            original_features,
            envelope_features,
            labels,
            epochs=30,
            batch_size=32,
            lr=0.01,
            use_scheduler=True,
            use_early_stopping=True,
            verbose=True):

        # Building the dataset
        train_dataset = DualSpectrumNetDataset(original_features, envelope_features, labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Setting the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)

        # Setting the scheduler
        if use_scheduler:
            scheduler = ReduceLROnPlateau(optimizer, "min", patience=10)

        # Setting the loss function
        criterion = nn.MSELoss()

        # Setting the early stopping mechanism
        if use_early_stopping:
            early_stopping = EarlyStopping("loss")

        # Training loop
        for epoch in range(epochs):

            # Epoch cumulated error
            train_loss = 0.0

            # For each minibatch...
            for (original_features, envelope_features), labels in tqdm(train_loader, disable=not verbose):

                # Applying the optimization
                predictions = self(original_features, envelope_features)
                loss = criterion(predictions, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if use_scheduler:
                    scheduler.step(loss)
                train_loss += loss.item()

            # Displaying the epoch average error
            if verbose:
                print(f"Epoch {epoch + 1} - train_loss: {train_loss:.4f}")

            # Assessing the early stopping
            if use_early_stopping:
                early_stopping(train_loss, self)
                if early_stopping.early_stop:
                    if verbose:
                        print("Training stopped by early stopping. Best accuracy: "
                              f"{early_stopping.best_score:.4f}")
                    break

        # Retrieving the model best parameters
        if use_early_stopping:
            self.load_state_dict(early_stopping.best_model_weights)

        return round(train_loss, 6)

    # Producing predictions
    def predict(self, original_features, envelope_features):
        self.eval()
        original_features = torch.from_numpy(original_features).float()
        envelope_features = torch.from_numpy(envelope_features).float()
        with torch.no_grad():
            predictions = self(original_features, envelope_features).cpu().numpy()
        return predictions

    # Saving the model and its metadata
    def save(self):
        with open(f"./model/model.pickle", "wb") as file:
            pickle.dump(self, file)
        metadata = {
            "signature": "dual_spectrum_net",
            "trained at": datetime.datetime.now().strftime("%Y-%m-%d %X")
        }
        with open(f"./model/metadata.json", "w") as file:
            json.dump(metadata, file, indent=4)