## PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection
by Kye-Hyeon Kim, Yeongjae Cheon, Sanghoon Hong, Byungseok Roh, Minje Park (Intel Imaging and Camera Technology)

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

### Citing PVANET

If you find PVANET useful in your research, please consider citing:

    @article{KimKH2016arXivPVANET,
        author = {Kye-Hyeon Kim and Yeongjae Cheon and Sanghoon Hong and Byungseok Roh and Minje Park},
        title = {{PVANET}: Deep but Lightweight Neural Networks for Real-time Object Detection},
        journal = {arXiv preprint arXiv:1608.08021},
        year = {2016}
    }

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
    #   Do NOT uncomment `USE_CUDNN := 1` (for running PVANET, cuDNN is slower than Caffe native implementation)
    #   Uncomment `WITH_PYTHON_LAYER := 1`

    make -j8 && make pycaffe
    ```

4. Download PVANET caffemodels
    ```Shell
    cd $FRCN_ROOT
    ./models/pvanet/download_models.sh
    ```

### Models

1. PVANET
- `./models/pvanet/full/test.pt`: For testing-time efficiency, batch normalization (w/ its moving averaged mini-batch statistics) and scale (w/ its trained parameters) layers are merged into the corresponding convolutional layer.
- `./models/pvanet/full/original.pt`: Original network structure.

2. PVANET (compressed)
- `./models/pvanet/comp/test.pt`: Compressed network w/ merging batch normalization and scale.
- `./models/pvanet/comp/original.pt`: Original compressed network structure.

3. PVANET-lite
- TBA


### How to run the demo

1. Download PASCAL VOC 2007 and 2012

Follow the instructions in [py-faster-rcnn README.md](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)

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

### Expected results

- PVANET+: 83.85% mAP
- PVANET+ (compressed): 82.90% mAP


