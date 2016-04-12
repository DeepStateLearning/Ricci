FROM continuumio/anaconda

RUN sed -i -e "s/wheezy/jessie/g" /etc/apt/sources.list

RUN apt-get update && apt-get upgrade && apt-get install -y g++ gcc g++-4.9 gcc-4.9 xorg libfftw3-dev

RUN pip install pyfftw

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
