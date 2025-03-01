#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Configuration parameters

"""

# Importing packages
import yaml

# Configuration parameter class
class Config:

    # Constructor
    def __init__(self):
        with open("./sources/config.yaml", "r") as file:
            config = yaml.safe_load(file)
        self.min_frequency = float(config["min_frequency"])
        self.precision = int(config["precision"])
        self.eps = float(config["eps"])
        self.feature_frequency_resolution = float(config["feature_frequency_resolution"])
        self.feature_max_frequency = float(config["feature_max_frequency"])
        self.model_path = str(config["model_path"])