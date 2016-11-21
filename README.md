## PVANET: Lightweight Deep Neural Networks for Real-time Object Detection
by Sanghoon Hong, Byungseok Roh, Kye-hyeon Kim, Yeongjae Cheon, Minje Park (Intel Imaging and Camera Technology)

### Introduction

This repository is a fork from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) and demonstrates the performance of PVANET.

You can refer to [py-faster-rcnn README.md](https://github.com/rbgirshick/py-faster-rcnn/blob/master/README.md) and [faster-rcnn README.md](https://github.com/ShaoqingRen/faster_rcnn/blob/master/README.md) for more information.

### Desclaimer

Please note that this repository doesn't contain our in-house runtime code used in the published article.
- The original py-faster-rcnn is quite slow and there exist lots of inefficient code blocks.
- We improved some of them, by 1) replacing the Caffe backend with its latest version (Sep 1, 2016), and 2) porting our implementation of the proposal layer.
- However it is still slower than our in-house runtime code due to the image pre-processing code written in Python (+9ms) and some poorly implemented parts in Caffe (+5 ms).
- PVANET was trained by our in-house deep learning library, not by this implementation.
- There might be a tiny difference in VOC2012 test results, because some hidden parameters in py-faster-rcnn may be set differently with ours.
- PVANET-lite (76.3% mAP on VOC2012, 10th place) is originally designed to verify the effectiveness of multi-scale features for object detection, so it only uses Inception and hyper features only. Further improvement may be achievable by adding C.ReLU, residual connections, etc.

### Citing PVANET

The BibTeX for EMDNN2016-accepted version will be updated soon

### Installation

1. Clone the Faster R-CNN repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/sanghoon/pva-faster-rcnn.git
  ```

2. We'll call the directory that you cloned Faster R-CNN into `FRCN_ROOT`. Build the Cython modules
    ```Shell
    cd $FRCN_ROOT/lib
    make
    ```

3. Build Caffe and pycaffe
    ```Shell
    cd $FRCN_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html
    # For your Makefile.config:
    #   Uncomment `WITH_PYTHON_LAYER := 1`

    cp Makefile.config.example Makefile.config
    make -j8 && make pycaffe
    ```

4. Download PVANET caffemodels
    ```Shell
    cd $FRCN_ROOT
    ./models/pvanet/download_models.sh
    ```
  - If it does not work,
    1. Download [full/test.model](https://drive.google.com/open?id=0BwFPOX3S4VcBd3NPNmI1RHBZNkk) and move it to ./models/pvanet/full/
    2. Download [comp/test.model](https://drive.google.com/open?id=0BwFPOX3S4VcBODJkckhudE1NeGM) and move it to ./models/pvanet/comp/

5. (Optional) Download original caffemodels (without merging batch normalization and scale layers)
    ```Shell
    cd $FRCN_ROOT
    ./models/pvanet/download_original_models.sh
    ```
  - If it does not work,
    1. Download [full/original.model](https://drive.google.com/open?id=0BwFPOX3S4VcBUW1OS1Fva3VKZ1E) and move it to ./models/pvanet/full/
    2. Download [comp/original.model](https://drive.google.com/open?id=0BwFPOX3S4VcBdVZuX3dQRzFjU1k) and move it to ./models/pvanet/comp/

6. (Optional) Download ImageNet pretrained models
    ```Shell
    cd $FRCN_ROOT
    ./models/pvanet/download_imagenet_models.sh
    ```
  - If it does not work,
    1. Download [imagenet/original.model](https://drive.google.com/open?id=0BwFPOX3S4VcBd1VtRzdHa1NoN1k) and move it to ./models/pvanet/imagenet/
    2. Download [imagenet/test.model](https://drive.google.com/open?id=0BwFPOX3S4VcBWnI0VHRzZWh6bFU) and move it to ./models/pvanet/imagenet/

7. (Optional) Download PVANET-lite models
    ```Shell
    cd $FRCN_ROOT
    ./models/pvanet/download_lite_models.sh
    ```
  - If it does not work,
    1. Download [lite/original.model](https://drive.google.com/open?id=0BwFPOX3S4VcBc1ZEQldZTlZKN00) and move it to ./models/pvanet/lite/
    2. Download [lite/test.model](https://drive.google.com/open?id=0BwFPOX3S4VcBSWg2MlpGcWlQeHM) and move it to ./models/pvanet/lite/

### Models

1. PVANET
  - `./models/pvanet/full/test.pt`: For testing-time efficiency, batch normalization (w/ its moving averaged mini-batch statistics) and scale (w/ its trained parameters) layers are merged into the corresponding convolutional layer.
  - `./models/pvanet/full/original.pt`: Original network structure.

2. PVANET (compressed)
  - `./models/pvanet/comp/test.pt`: Compressed network w/ merging batch normalization and scale.
  - `./models/pvanet/comp/original.pt`: Original compressed network structure.

3. PVANET (ImageNet pretrained model)
  - `./models/pvanet/imagenet/test.pt`: Classification network w/ merging batch normalization and scale.
  - `./models/pvanet/imagenet/original.pt`: Original classification network structure.

4. PVANET-lite
  - `./models/pvanet/lite/test.pt`: Compressed network w/ merging batch normalization and scale.
  - `./models/pvanet/lite/original.pt`: Original compressed network structure.


### How to run the demo

1. Download PASCAL VOC 2007 and 2012
  - Follow the instructions in [py-faster-rcnn README.md](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)

2. PVANET+ on PASCAL VOC 2007
  ```Shell
  cd $FRCN_ROOT
  ./tools/test_net.py --gpu 0 --def models/pvanet/full/test.pt --net models/pvanet/full/test.model --cfg models/pvanet/cfgs/submit_160715.yml
  ```

3. PVANET+ (compressed)
  ```Shell
  cd $FRCN_ROOT
  ./tools/test_net.py --gpu 0 --def models/pvanet/comp/test.pt --net models/pvanet/comp/test.model --cfg models/pvanet/cfgs/submit_160715.yml
  ```

4. (Optional) ImageNet classification
  ```Shell
  cd $FRCN_ROOT
  ./caffe-fast-rcnn/build/tools/caffe test -gpu 0 -model models/pvanet/imagenet/test.pt -weights models/pvanet/imagenet/test.model -iterations 1000
  ```

5. (Optional) PVANET-lite
  ```Shell
  cd $FRCN_ROOT
  ./tools/test_net.py --gpu 0 --def models/pvanet/lite/test.pt --net models/pvanet/lite/test.model --cfg models/pvanet/cfgs/submit_160715.yml
  ```

### Expected results

- PVANET+: 83.85% mAP
- PVANET+ (compressed): 82.90% mAP
- ImageNet classification: 68.998% top-1 accuracy, 88.8902% top-5 accuracy, 1.28726 loss
- PVANET-lite: 79.10% mAP
