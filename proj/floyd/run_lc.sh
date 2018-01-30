#!/bin/bash
# sudo -b nohup nvidia-docker-plugin > /tmp/nvidia-docker.log
nvidia-docker run -it -p 9902:8888 -d -v $PWD:/notebooks -v /raid/wgts/pt/weights:/usr/local/lib/python3.6/site-packages/fastai-0.6-py3.6.egg/fastai/weights -v /raid/system/dbi:/data fastai:dev
