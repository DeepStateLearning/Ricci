#!/bin/bash
docker-machine stop default
VBoxManage modifyvm default --cpus 4
VBoxManage modifyvm default --memory 8192
docker-machine start default
