# syntax=docker/dockerfile:1

FROM python:3.10-bullseye

RUN apt-get update; \ 
    apt-get install -y --no-install-recommends \
    libsndfile1

EXPOSE 7866

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .


CMD ["python3", "infer-web.py"]