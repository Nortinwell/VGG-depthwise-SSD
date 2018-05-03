#!/bin/bash
GPU_ID=0
cur_date=`date +%Y-%m-%d-%H-%M-%S`
log_file_name=/home/ubuntu/Ding/SSD/caffe/examples/ssd/logs/${cur_date}
./build/tools/caffe train \
    -solver /home/ubuntu/Ding/SSD/caffe/examples/ssd/solver.prototxt \
  #  -weights /home/ubuntu/Ding/SSD/caffe/examples/ssd/models/VGGNet/VOC0712/SSD_300x300_iter_20000.caffemodel \
    -gpu $GPU_ID 2>&1 | tee -a ${log_file_name}
