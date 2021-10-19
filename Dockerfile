FROM ubuntu:latest

USER root

RUN apt-get update
RUN apt-get install build-essential -y
RUN apt-get install linux-tools-generic -y
RUN apt-get install make -y
RUN apt-get install wget tar sudo -y
RUN wget https://cdn.kernel.org/pub/linux/kernel/v4.x/linux-4.14.173.tar.xz && tar -xf ./linux-4.14.173.tar.xz && cd linux-4.14.173/tools/perf/ && apt -y install flex bison && make -C . && make install
# RUN apt-get install mpich -y 

USER root

RUN mkdir /parallel
WORKDIR /parallel
COPY ./ /parallel

# docker build -t parallel . && docker run --rm parallel sh -c "make mpi && make run_mpi"