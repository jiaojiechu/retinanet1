import os
import os
import random
from PIL import Image
import xml.etree.ElementTree as ET
class UnsupportedExtensionError(Exception):
    def __init__(self, ext):
        message = '{} is not a known file extension'.format(ext)
        super(UnsupportedExtensionError, self).__init__(message)
class UnsupportedFormatError(Exception):
    def __init__(self, fmt):
        message = '{} is not a known annotation format'.format(fmt)
        super(UnsupportedFormatError, self).__init__(message)
class BoundingBox:
    def __init__(self, left, top, right, bottom, image_width, image_height, label):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.width = right - left
        self.height = bottom - top
        self.image_width = image_width
        self.image_height = image_height
        self.label = label

    def __repr__(self):
        return '(x1: {}, y1: {}, x2: {}, y2: {} ({}))'.format(self.left, self.top, self.right, self.bottom, self.label)

    def flip(self):
        left = self.image_width - self.right
        top = self.image_height - self.bottom
        right = self.image_width - self.left
        bottom = self.image_height - self.top
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        return self

    def resize(self, width, height):
        width_ratio = width / self.image_width
        height_ratio = height / self.image_height
        self.left = int(self.left * width_ratio)
        self.top = int(self.top * height_ratio)
        self.right = int(self.right * width_ratio)
        self.bottom = int(self.bottom * height_ratio)
        return self
class AnnotationDir:
    supported_exts = ['.xml']
    supported_formats = ['voc']

    def __init__(self, path, filenames, labels, ext, fmt):
        if not ext in AnnotationDir.supported_exts:
            raise UnsupportedExtensionError(ext)
        if not fmt in AnnotationDir.supported_formats:
            raise UnsupportedFormatError(fmt)
        self.path = path
        self.filenames = ['{}{}'.format(fn, ext) for fn in filenames]
        self.labels = labels
        self.fmt = fmt
        self.ann_dict = self.build_annotations()

    def build_annotations(self):
        box_dict = {}
        if self.fmt == 'voc':
            for fn in self.filenames:
                boxes = []
                tree = ET.parse(os.path.join(self.path, fn))
                ann_tag = tree.getroot()

                size_tag = ann_tag.find('size')
                image_width = int(size_tag.find('width').text)
                image_height = int(size_tag.find('height').text)

                for obj_tag in ann_tag.findall('object'):
                    label = obj_tag.find('name').text

                    box_tag = obj_tag.find('bndbox')
                    left = int(box_tag.find('xmin').text)
                    top = int(box_tag.find('ymin').text)
                    right = int(box_tag.find('xmax').text)
                    bottom = int(box_tag.find('ymax').text)

                    box = BoundingBox(left, top, right, bottom, image_width, image_height, self.labels.index(label))
                    boxes.append(box)
                if len(boxes) > 0:
                    box_dict[os.path.splitext(fn)[0]] = boxes
                else:
                    self.filenames.remove(fn)
        return box_dict

    def get_boxes(self, fn):
        return self.ann_dict[fn]
root='E:\\RETINANET\\retinanet.pytorch-master\\retinanet.pytorch-master\\data2\\VOC2007'
image_dir = os.path.join(root, 'JPEGImages')
annotation_dir = os.path.join(root, 'Annotations')
train_imageset_fn = os.path.join(root, 'ImageSets', 'Main', 'vehicle1.txt')
imageset_fn=train_imageset_fn
#val_imageset_fn = os.path.join(root, 'ImageSets', 'Main', 'val.txt')
image_ext = '.jpg'
image_dir_path = image_dir
image_ext = image_ext
classes = ['vehicle1']
with open(imageset_fn) as f:
            #rstrip()删除指定的字符。默认为空格
    filenames = [fn.rstrip() for fn in f.readlines()]
annotation_dir = AnnotationDir(annotation_dir, filenames, classes, '.xml', 'voc')
#filenames = list(annotation_dir.ann_dict.keys())
filenames=[str(i) for i in range(855)]
f=open('E:\\RETINANET\\retinanet.pytorch-master\\retinanet.pytorch-master\\voc2007train.txt','w+')
for fn in filenames:
    image_fn = '{}{}'.format(fn, image_ext)
    image_path = os.path.join(image_dir_path, image_fn)
    image = Image.open(image_path)
    boxes = annotation_dir.get_boxes(fn)
    example = {'image': image, 'boxes': boxes}
    vv=[]
    for i in boxes:
        vv.append(i.left)
        vv.append(i.top)
        vv.append(i.right)
        vv.append(i.top+i.height)
        vv.append(i.label)
    f.write(image_fn)
    for i in vv:
        f.write(' ')
        f.write(str(i))
    f.write('\r\n')

    print(example)
f.close()

#self.annotation_dir = AnnotationDir(annotation_dir, self.filenames, classes, '.xml', 'voc')