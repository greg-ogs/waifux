#!/usr/bin/bash

# Isntall java JDK
apt-get update && apt-get install -y openjdk-17-jdk
apt-get install cmake build-essential
apt-get install libportaudio2

# Install python dependencies
python3 -m pip install --upgrade pip
pip install -r requirements.txt
# pip install Flask --ignore-installed blinker
