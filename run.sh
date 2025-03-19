#!/bin/bash
cd /home/user/fastapi-api
docker stop fastapi-api || true
docker rm fastapi-api || true
docker build -t fastapi-api .
docker run -d --restart always -p 8000:8000 fastapi-api
