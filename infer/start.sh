#!/usr/bin/env bash

export FLASK_APP=app.py
export LD_PRELOAD=./plugin/libnvinfer_plugin.so

gunicorn --workers 1 -k sync --timeout 300 --graceful-timeout 600 --bind 0.0.0.0:8080 app:app