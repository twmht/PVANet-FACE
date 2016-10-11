## PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection
by Kye-Hyeon Kim, Sanghoon Hong, Byungseok Roh, Yeongjae Cheon, Minje Park (Intel Imaging and Camera Technology)

### Notes
- The training of PVANet 9.0 on the VOC2012 leaderboard wasn't done with this code.
- PVANet pre-trained model contains a FC6 layer with a 6x6x384-shaped input. Therefore, layers in this training example generates **a hyper feature with the depth of 384** which is different from the one in the arXiv article.
- For better detection results, **fine-tuning the existing PVANet** is recommended (see example_finetune).

### Sample command
- Training for 100k iterations (toy)
    ```
    tools/train_net.py 
        --gpu 0
        --solver models/pvanet/example_train_384/solver.prototxt
        --weights models/pvanet/imagenet/original.model
        --iters 100000
        --cfg models/pvanet/cfgs/train.yml
        --imdb voc_2007_trainval
    ```

- Testing

    ```
    tools/test_net.py
        --gpu 0
        --def models/pvanet/example_train_384/test.prototxt
        --net output/faster_rcnn_pvanet/voc_2007_trainval/pvanet_frcnn_384_iter_100000.caffemodel
        --cfg models/pvanet/cfgs/submit_160715.yml 
    ```

