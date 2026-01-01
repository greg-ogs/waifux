#!/usr/bin/bash

# Isntall java JDK
apt-get update && apt-get install -y openjdk-17-jdk

# Install python dependencies
python3 -m pip install --upgrade pip
pip install -r requirements.txt
pip uninstall tensorflow -y
pip install --upgrade "tensorflow[and-cuda]==2.20.0"
pip uninstall protobuf -y
pip install "protobuf<6.0.0, >=5.28.3"
# pip install Flask --ignore-installed blinker
