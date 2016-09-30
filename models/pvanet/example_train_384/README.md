## PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection
by Kye-Hyeon Kim, Sanghoon Hong, Byungseok Roh, Yeongjae Cheon, Minje Park (Intel Imaging and Camera Technology)

### Notes
- The training of PVANet 9.0 on the VOC2012 leaderboard wasn't done with this code.
- **This training prototxt hasn't been tested yet**.
- PVANet pre-trained model contains a FC6 layer with a 6x6x384-shaped input. Therefore, layers in this training example generates **a hyper feature with the depth of 384** which is different from the one in the arXiv article.

