FROM floydhub/pytorch:0.3.0-gpu.cuda9cudnn7-py3.22
MAINTAINER Kevin
COPY floyd_requirements.txt floyd_requirements.txt 
RUN ["pip3","--default-timeout=300 ", "install","-r","floyd_requirements.txt","-i","https://pypi.tuna.tsinghua.edu.cn/simple"]
COPY fastai fastai
RUN cd fastai && python3 setup.py install
LABEL version="1.0" description="This image is used to set up fastai dev env."


