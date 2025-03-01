#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Guessing the RPM of a signal from the original and envelope spectrum features

"""

# Importing packages and modules
import pickle

# Guesser class
class Guesser:

    # Constructor
    def __init__(self):
        pass

    #  Computing the FFT
    def guess(self, original_features: list, envelope_features: list):
        guessed_rpm = 1200
        return guessed_rpm