FROM pytorch/pytorch:latest

MAINTAINER Auguste Gardette <auguste.gardette@hotmail.fr>

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt