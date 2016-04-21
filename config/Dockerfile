FROM continuumio/anaconda

RUN sed -i -e "s/wheezy/jessie/g" /etc/apt/sources.list

RUN echo "deb http://llvm.org/apt/jessie/ llvm-toolchain-jessie-3.8 main" >>  /etc/apt/sources.list && \
    wget -O - http://llvm.org/apt/llvm-snapshot.gpg.key | apt-key add -

RUN apt-get update && apt-get upgrade -y && apt-get install -y g++-4.9 gcc-4.9 gcc g++ clang-3.8 xorg libfftw3-dev make

RUN pip install pyfftw

ENV LIBRARY_PATH ".:/opt/conda/lib"
ENV LD_LIBRARY_PATH ".:/opt/conda/lib"
ENV CPATH "/usr/lib/gcc/x86_64-linux-gnu/4.8/include"
ENV OMP_DYNAMIC FALSE
ENV OMP_PROC_BIND TRUE
ENV CC clang-3.8
ENV CXX clang++-3.8
ENV LIBOMP "-liomp5"

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
