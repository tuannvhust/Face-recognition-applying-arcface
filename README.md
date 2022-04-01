# Face-recognition-applying-arcface
train data : https://onedrive.live.com/?authkey=%21AJjQxHY%2DaKK%2DzPw&cid=1FD95D6F0AF30F33&id=1FD95D6F0AF30F33%2174855&parId=1FD95D6F0AF30F33%2174853&action=locate

test data : http://vis-www.cs.umass.edu/lfw/

###Training
CUDA_VISIBLE_DEVICES=0 python3 main.py --scenario='train'

####Visualize
tensorboard --logdir [logs direction]
rồi click vào link
