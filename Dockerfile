FROM sheercat/fbprophet:latest
COPY sources.list /etc/apt/
# python dependency
RUN mkdir ~/.pip
COPY pip.conf ~/.pip/pip.conf
COPY requirements.txt requirements.txt
RUN ["pip3","--default-timeout=300 ", "install","-r","requirements.txt","-i","https://pypi.tuna.tsinghua.edu.cn/simple"]
