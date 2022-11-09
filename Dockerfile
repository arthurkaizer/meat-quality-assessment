from python:3.10.8-bullseye

label maintainer="natancarmo321@gmail.com"

expose 5000

run apt update
run apt install ffmpeg libsm6 libxext6 -y

workdir /ti6-api

run pip install --no-cache-dir --upgrade pip

copy requirements.txt .

run pip install -r requirements.txt

copy . .

cmd ["python", "src/main.py"]
