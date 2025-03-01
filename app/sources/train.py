#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

RPM Deep Guesser model training

"""

# Importing packages and modules
import numpy as np
from dual_spectrum_net import DualSpectrumNet

#  Running the training pipeline
if __name__ == "__main__":

    # Hiperparameters
    batch_size = 32
    number_samples = 128
    number_features = 1000

    # Generating random data
    original_features = np.random.rand(number_samples, number_features).astype(np.float32)
    envelope_features = np.random.rand(number_samples, number_features).astype(np.float32)
    labels = np.random.normal(1200, 400, size=(number_samples,))

    # Instantiating and training a DSN model
    dsn = DualSpectrumNet(number_features, [300, 200, 100])
    dsn.fit(original_features, envelope_features, labels)
    dsn.save()