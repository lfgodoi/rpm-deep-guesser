#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Extracting features from original signal and envelope spectra to be processed by the deep learning model

"""

# Importing packages and modules
import numpy as np
from sources.config import Config

# Feature Extractor class
class FeatureExtractor:

    # Constructor
    def __init__(self):

        # Configuration parameters
        config = Config()
        self.feature_frequency_resolution = config.feature_frequency_resolution
        self.feature_max_frequency = config.feature_max_frequency

        # Instance parameters
        self.confidence_margin = config.eps
        self.feature_frequencies = np.arange(0, self.feature_max_frequency, self.feature_frequency_resolution)

    # Updating the confidence margin according to the current feature extraction
    def update_confidence_margin(self, frequencies):
        resolution = frequencies[1] - frequencies[0]
        if resolution > self.confidence_margin/60:
            self.confidence_margin = resolution*60

    #  Computing the FFT
    def extract(self, amplitudes: list, frequencies: list):
        features = []
        for feature_frequency in self.feature_frequencies:
            index = np.argmin(np.abs(frequencies - feature_frequency))
            features.append(float(amplitudes[index]))
        features = np.array(features)
        features = (features - features.min())/(features.max() - features.min())
        self.update_confidence_margin(frequencies)
        return features