# A *Faster* Pytorch Implementation of Faster R-CNN

## Introduction

This project is a *faster* pytorch implementation of faster R-CNN, aimed to accelerating the training of faster R-CNN object detection models. Recently, there are a number of good implementations:


* [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), developed based on Pytorch + TensorFlow + Numpy

During our implementing, we referred the above implementations, especailly [longcw/faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch). However, our implementation has several unique and new features compared with the above implementations:

* **It is pure Pytorch code**. We convert all the numpy implementations to pytorch!

* **It supports multi-image batch training**. We revise all the layers, including dataloader, rpn, roi-pooling, etc., to support multiple images in each minibatch.

* **It supports multiple GPUs training**. We use a multiple GPU wrapper (nn.DataParallel here) to make it flexible to use one or more GPUs, as a merit of the above two features.

* **It supports three pooling methods**. We integrate three pooling methods: roi pooing, roi align and roi crop. More importantly, we modify all of them to support multi-image batch training.

* **It is memory efficient**. We limit the image aspect ratio, and group images with similar aspect ratios into a minibatch. As such, we can train resnet101 and VGG16 with batchsize = 4 (4 images) on a sigle Titan X (12 GB). When training with 8 GPU, the maximum batchsize for each GPU is 3 (Res101), totally 24.

* **It is faster**. Based on the above modifications, the training is much faster. We report the training speed on NVIDIA TITAN Xp in the tables below.

### What we are doing and going to do

- [x] Support both python2 and python3 (great thanks to [cclauss](https://github.com/cclauss)).
- [x] Add deformable pooling layer (mainly supported by [Xander](https://github.com/xanderchf)).
- [x] Support pytorch-0.4.0.
- [x] Support tensorboardX.
- [ ] Support pytorch-0.4.1 or higher.



* [Blog](http://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/) by [ankur6ue](https://github.com/ankur6ue)

