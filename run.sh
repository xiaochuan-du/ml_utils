#!/bin/bash
# sudo -b nohup nvidia-docker-plugin > /tmp/nvidia-docker.log
nvidia-docker run -it -p 8888:8888 -d -v $PWD:/notebooks -v /raid/wgts:/root/.keras/models -v /raid/system/bld:/data tf:gpu
