FROM continuumio/anaconda

RUN sed -i -e "s/wheezy/jessie/g" /etc/apt/sources.list

RUN apt-get update && apt-get upgrade -y && apt-get install -y g++ gcc g++-4.9 gcc-4.9 xorg libfftw3-dev make

RUN pip install pyfftw

ENV LIBRARY_PATH "."
ENV LD_LIBRARY_PATH "."
ENV OMP_DYNAMIC FALSE
ENV OMP_PROC_BIND TRUE

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
