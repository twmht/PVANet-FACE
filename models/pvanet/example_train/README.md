## PVANet: Lightweight Deep Neural Networks for Real-time Object Detection
by Sanghoon Hong, Byungseok Roh, Kye-hyeon Kim, Yeongjae Cheon, Minje Park (Intel Imaging and Camera Technology)
Presented in [EMDNN2016](http://allenai.org/plato/emdnn/), a NIPS2016 workshop ([arXiv link](https://arxiv.org/abs/1611.08588))

### Notes
- The training of PVANet 9.1 on the VOC2012 leaderboard wasn't done with this code.

### Sample command
- Training for 100k iterations (toy)
    ```
    tools/train_net.py 
        --gpu 0
        --solver models/pvanet/example_train/solver.prototxt
        --weights models/pvanet/pretrained/pva9.1_pretrained_no_fc6.caffemodel
        --iters 100000
        --cfg models/pvanet/cfgs/train.yml
        --imdb voc_2007_trainval
    ```

- Testing

    ```
    tools/test_net.py
        --gpu 0
        --def models/pvanet/example_train/test.prototxt
        --net output/faster_rcnn_pvanet/voc_2007_trainval/pvanet_frcnn_iter_100000.caffemodel
        --cfg models/pvanet/cfgs/submit_160715.yml 
    ```
