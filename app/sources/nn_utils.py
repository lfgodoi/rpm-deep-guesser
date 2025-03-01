#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Neural network utilities

"""

# Importing packages and modules

import copy
import math

# Early stopping algorithm
class EarlyStopping:

    # Constructor
    def __init__(self, mode="accuracy", patience=10):
        self.mode = mode
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.best_model_weights = None
        if mode == "accuracy":
            self.best_score = 0
        elif mode == "loss":
            self.best_score = math.inf

    # Early stopping criteria assessment
    def __call__(self, score, model):
        to_update = False
        if self.mode == "accuracy":
            to_update = score > self.best_score
        elif self.mode == "loss":
            to_update = score < self.best_score
        if to_update:
            self.best_score = score
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.counter = self.patience
        else:
            self.counter -= 1
            if self.counter == 0:
                self.early_stop = True