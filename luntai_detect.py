# coding: utf-8
# Note: this file is expected to be in {caffe_root}/examples
# ### 1. Setup
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pylab
import time
from nms import nms

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe_root = '../'
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, '/home/ubuntu/Ding/SSD/caffe/python')
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2

caffe.set_device(0)
#caffe.set_mode_gpu()
labelmap_file = '/home/ubuntu/Ding/SSD/caffe/examples/ssd/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


model_def = '/home/ubuntu/Ding/SSD/caffe/examples/ssd/deploy.prototxt'
model_weights = '/home/ubuntu/Ding/SSD/caffe/examples/ssd/models/VGGNet/VOC0712/VGG-reduce_SSD_150+BN+msra.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
transformer.set_raw_scale(
    'data', 255
)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap(
    'data',
    (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

# ### 2. SSD detection

# Load an image.

image_resize = 300
net.blobs['data'].reshape(1, 3, image_resize, image_resize)

image = caffe.io.load_image('/home/ubuntu/Ding/SSD/caffe/examples/images/8.png')
plt.imshow(image)

# Run the net and examine the top_k results

transformed_image = transformer.preprocess('data', image)
net.blobs['data'].data[...] = transformed_image

# Forward pass.
start=time.time() # time begin
detections = net.forward()['detection_out']
use_time=time.time()-start # proc time
print("time = "+str(use_time)+" s")

# Parse the outputs.
det_label = detections[0, 0, :, 1]
det_conf = detections[0, 0, :, 2]
det_xmin = detections[0, 0, :, 3]
det_ymin = detections[0, 0, :, 4]
det_xmax = detections[0, 0, :, 5]
det_ymax = detections[0, 0, :, 6]

# Get detections with confidence higher than 0.6.
top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.3]

top_conf = det_conf[top_indices]
top_label_indices = det_label[top_indices].tolist()
top_labels = get_labelname(labelmap, top_label_indices)
top_xmin = det_xmin[top_indices]
top_ymin = det_ymin[top_indices]
top_xmax = det_xmax[top_indices]
top_ymax = det_ymax[top_indices]

# Plot the boxes

colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

currentAxis = plt.gca()

for i in xrange(top_conf.shape[0]):
    # bbox value
    xmin = int(round(top_xmin[i] * image.shape[1]))
    ymin = int(round(top_ymin[i] * image.shape[0]))
    xmax = int(round(top_xmax[i] * image.shape[1]))
    ymax = int(round(top_ymax[i] * image.shape[0]))
    # score
    score = top_conf[i]
    # label
    label = int(top_label_indices[i])
    label_name = top_labels[i]
    # display info: label score xmin ymin xmax ymax
    display_txt = '%s: %.2f' % (label_name, score)
    # display_bbox_value = '%d %d %d %d' % (xmin, ymin, xmax, ymax)
    coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
    color = colors[label]
    currentAxis.add_patch(
        plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
    currentAxis.text(
        xmin, ymin, display_txt, bbox={'facecolor': color,
                                       'alpha': 0.5})
    # currentAxis.text((xmin+xmax)/2, (ymin+ymax)/2, display_bbox_value, bbox={'facecolor': color, 'alpha': 0.5})
plt.imshow(image)
pylab.show()
