#/bin/bash

# ImageNet pre-trained
wget https://www.dropbox.com/s/a2y0e12kmu8wjsf/pva9.1_preAct_train_iter_1900000.caffemodel?dl=1 -O models/pvanet/pretrained/pva9.1_preAct_train_iter_1900000.caffemodel

# COCO train/val + VOC0712 train/val
wget https://www.dropbox.com/s/sq1kujjil5qg5bw/PVA9.1_ImgNet_COCO_VOC0712.caffemodel?dl=1 -O models/pvanet/pva9.1/PVA9.1_ImgNet_COCO_VOC0712.caffemodel

# COCO train/val + VOC0712 train/val + VOC07 test
wget https://www.dropbox.com/s/m65ioxcguevsacc/PVA9.1_ImgNet_COCO_VOC0712plus.caffemodel?dl=1 -O models/pvanet/pva9.1/PVA9.1_ImgNet_COCO_VOC0712plus.caffemodel

# COCO train/val + VOC0712 train/val + VOC07 test + network compression
wget https://www.dropbox.com/s/76q7pdym70ji986/PVA9.1_ImgNet_COCO_VOC0712plus_compressed.caffemodel?dl=1 -O models/pvanet/pva9.1/PVA9.1_ImgNet_COCO_VOC0712plus_compressed.caffemodel