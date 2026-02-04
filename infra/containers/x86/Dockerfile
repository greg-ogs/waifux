FROM nvcr.io/nvidia/pytorch:26.01-py3
LABEL authors="greg"

WORKDIR /app

COPY requirements.txt startup.sh /app/

RUN chmod +x startup.sh && sh ./startup.sh

RUN rm startup.sh requirements.txt