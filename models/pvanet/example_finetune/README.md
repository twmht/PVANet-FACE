## PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection
by Kye-Hyeon Kim, Sanghoon Hong, Byungseok Roh, Yeongjae Cheon, Minje Park (Intel Imaging and Camera Technology)

### Notes
- The training of PVANet 9.0 on the VOC2012 leaderboard wasn't done with this code.

### Sample command
- Training for 100k iterations (toy)
    ```
    tools/train_net.py 
        --gpu 0
        --solver models/pvanet/example_finetune/solver.prototxt
        --weights models/pvanet/full/test.model
        --iters 100000
        --cfg models/pvanet/cfgs/train.yml
        --imdb voc_2007_trainval
    ```

- Testing

    ```
    tools/test_net.py
        --gpu 0
        --def models/pvanet/example_finetune/test.prototxt
        --net output/faster_rcnn_pvanet/voc_2007_trainval/pvanet_frcnn_iter_100000.caffemodel
        --cfg models/pvanet/cfgs/submit_160715.yml 
    ```
