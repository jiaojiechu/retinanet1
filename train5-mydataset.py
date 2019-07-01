import argparse
import os
import sys
import  tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import voc.transforms as transforms
from encoder import DataEncoder
from loss import FocalLoss
from retinanet import RetinaNet
from voc.datasets import VocLikeDataset
import  exps.voc.config2 as cfg
try:
    import ipdb
except:
    import pdb as ipdb
import  numpy as np

#import config as cfg

#assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')
start_epoch = 0
lr = cfg.lr

print('Preparing data..')
train_transform_list = [transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)]
if cfg.scale is not None:
    train_transform_list.insert(0, transforms.Scale(cfg.scale))
train_transform = transforms.Compose(train_transform_list)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cfg.mean, cfg.std)
])

trainset = VocLikeDataset(image_dir=cfg.image_dir, annotation_dir=cfg.annotation_dir, imageset_fn=cfg.train_imageset_fn,
                          image_ext=cfg.image_ext, classes=cfg.classes, encoder=DataEncoder(), transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True,
                                          num_workers=cfg.num_workers, collate_fn=trainset.collate_fn)

#ii=enumerate(trainloader)
#ipdb.set_trace()
#print(ii)
print('Building model...')
net = RetinaNet(backbone=cfg.backbone, num_classes=len(cfg.classes))
criterion = FocalLoss(len(cfg.classes))
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
#special_layers=nn.ModuleList(net.subnet_classes)
#special_layers_params=list(map(id,special_layers.parameters()))
#base_params=filter(lambda p:id(p) not in special_layers_params ,net.parameters())
#optimizer=torch.optim.SGD([{'params':base_params},{'params':special_layers_params.parameters(),'lr':0.001}],lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

def save_checkpoint(loss,net, n):
    global best_loss
    loss /= n
    if loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'loss': loss,
            'epoch': epoch,
            'lr': lr
        }
        #ckpt_path = os.path.join('ckpts', args.exp)
        ckpt_path='.store'
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path)
        torch.save(state, os.path.join(ckpt_path, 'ckpt3.pth'))
        best_loss = loss
ckpt_path='.store'
map_location = lambda storage, loc: storage

#net.load_state_dict(torch.load('.store\\ckpt.pth')['net'])
net.load_state_dict(torch.load('.store/ckpt3.pth',map_location=map_location)['net'],strict=False)
#lr=0.01
#net.load_state_dict(torch.load( '.store//ckpt.pth')['net'])
for epoch in range(start_epoch + 1, start_epoch + cfg.num_epochs + 1):
    if epoch in cfg.lr_decay_epochs:
        lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    print('\nTrain Epoch: %d' % epoch)
    net.train()

    train_loss = 0

    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        #print(np.any(np.isnan(inputs.numpy())))
        #print(np.any(np.isnan(loc_targets.numpy())))
        #print(np.any(np.isnan(loc_targets.numpy())))
        #ipdb.set_trace()
        pos = cls_targets > 0
        #pos1=cls_targets ==0
        #pos2=cls_targets ==-1

        print(pos.data.long().sum())
        #print(pos1.data.long().sum())
        #print(pos2.data.long().sum())
        inputs = Variable(inputs)
        loc_targets = Variable(loc_targets)
        cls_targets = Variable(cls_targets)
        #print(np.any(np.isnan(inputs.numpy())))
        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)

        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        #nn.utils.clip_grad_norm(net.parameters(), max_norm=1.2)
        optimizer.step()

        # train_loss += loss.data[0]
        train_loss += loss.data
        # print('train_loss: %.3f | avg_loss: %.3f' % (loss.data[0], train_loss/(batch_idx+1)))
        #if torch.isnan(loss.data):
        #    ipdb.set_trace()
         #   loc, cls = net(inputs)
        print('train_loss: %.3f | avg_loss: %.3f' % (loss.data, train_loss / (batch_idx + 1)))
    save_checkpoint(train_loss, net, len(trainloader))


