FROM ubuntu:22.04

RUN apt-get update && apt-get -y install python3 python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install poetry

RUN mkdir /app
WORKDIR /app

COPY models/ /models/
