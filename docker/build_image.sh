#!/usr/bin/env bash
nvidia-docker run -it -v /data:/data -p 8019:22 template:v0 /bin/bash


