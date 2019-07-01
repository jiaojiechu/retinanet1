import torch
import torchvision.transforms as transforms
import  exps.voc.config2 as cfg
from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw
try:
    import ipdb
except:
    import pdb as ipdb

print('Loading model..')
net = RetinaNet()
net = RetinaNet(backbone=cfg.backbone, num_classes=len(cfg.classes))
map_location = lambda storage, loc: storage

#net.load_state_dict(torch.load('.store\\ckpt.pth')['net'])
net.load_state_dict(torch.load('.store/ckpt.pth',map_location=map_location)['net'],strict=False)

print(torch.load('.store/ckpt.pth',map_location=map_location)['lr'])
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

print('Loading image..')
img = Image.open('./image/0.jpg')
w = 600
h=600
img = img.resize((w,h))
#ipdb.set_trace()
print('Predicting..')
x = transform(img)
x = x.unsqueeze(0)
x = Variable(x, volatile=True)
loc_preds, cls_preds = net(x)

print('Decoding..')
encoder = DataEncoder()
boxes, labels = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w,h))

draw = ImageDraw.Draw(img)
for box in boxes:
    draw.rectangle(list(box), outline='red')
img.show()
