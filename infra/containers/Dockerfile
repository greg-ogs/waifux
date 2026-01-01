FROM tensorflow/tensorflow:latest-gpu
LABEL authors="greg-ogs"

WORKDIR /app

COPY requirements.txt startup.sh /app/

RUN chmod +x startup.sh && sh ./startup.sh

RUN rm startup.sh requirements.txt