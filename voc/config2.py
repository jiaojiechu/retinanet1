import os

#root = os.path.join('data', 'VOC2007')
root='E:\\RETINANET\\retinanet.pytorch-master\\retinanet.pytorch-master\\data2\\VOC2007'
image_dir = os.path.join(root, 'JPEGImages')
annotation_dir = os.path.join(root, 'Annotations')
train_imageset_fn = os.path.join(root, 'ImageSets', 'Main', 'vehicle1.txt')
#val_imageset_fn = os.path.join(root, 'ImageSets', 'Main', 'val.txt')
image_ext = '.jpg'

backbone = 'resnet34'
#classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
#           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
classes = ['vehicle1']
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
scale = None

batch_size = 1
lr = 0.0001
momentum = 0.9
weight_decay = 0.01
num_epochs = 124
lr_decay_epochs = [83, 110]
num_workers =3

eval_while_training = False
eval_every = 10
#for i in  self.layer1.parameters():
 #   print(i)