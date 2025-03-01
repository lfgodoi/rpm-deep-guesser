#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Main script

"""

# Importing packages and modules
from flask import Flask
from routes.guessing import guessing_blueprint

# Setting the app
app = Flask(__name__)
app.config["VERSION"] = "1.3.0"

# Registering the blueprints
app.register_blueprint(guessing_blueprint)

# Running the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5016, debug=False)
