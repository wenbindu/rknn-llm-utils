#!/bin/bash

# prepare
sudo chown -R proembed:proembed /home/proembed/emotion
sudo mkdir -p /home/proembed/emotion/logs
sudo chmod 755 /home/proembed/emotion/logs

sudo cp sentiment-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sentiment-api
sudo systemctl start sentiment-api

sudo systemctl status sentiment-api
