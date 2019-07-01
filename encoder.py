import torch
from math import sqrt
from utils import box_iou, box_nms, change_box_order, meshgrid


class DataEncoder:
    def __init__(self):
        #self.anchor_areas = [512*512, 256*256, 128*128, 64*64, 32*32]
        self.anchor_areas = [100* 100 ,80*80,60 * 60,50*50,40*40 ]
        self.aspect_ratios = [1/2, 1, 2/1]
        self.scale_ratios = [1, pow(2, 1/3), pow(2, 2/3)]
        self.num_levels = len(self.anchor_areas)
        self.num_anchors = len(self.aspect_ratios) * len(self.scale_ratios)
        self.anchor_edges = self.calc_anchor_edges()

    def calc_anchor_edges(self):
        anchor_edges = []
        #得到5个不同面积下的9个候选框
        for area in self.anchor_areas:
            for ar in self.aspect_ratios:
                if ar < 1:
                    height = sqrt(area)
                    width = height * ar
                else:
                    width = sqrt(area)
                    height = width / ar
                for sr in self.scale_ratios:
                    anchor_height = height * sr
                    anchor_width = width * sr
                    anchor_edges.append((anchor_width, anchor_height))
        #5*9*2 变为（5,9,2）
        return torch.Tensor(anchor_edges).view(self.num_levels, self.num_anchors, 2)

    def get_anchor_boxes(self, input_size):
        #5个级别 图片缩小为8,16，32,64,128
        fm_sizes = [(input_size / pow(2, i + 3)).ceil() for i in range(self.num_levels)]

        boxes = []
        for i in range(self.num_levels):
            #每个级别对应一种尺寸大小
            fm_size = fm_sizes[i]
            #得到缩小倍数
            grid_size = (input_size / fm_size).floor()
            #缩小后的图片尺寸
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            #将每个图片每个点的坐标保存下来
            xy = meshgrid(fm_w, fm_h) + 0.5  # [fm_h * fm_w, 2]

            xy=xy.float()
            #再每个像素点对应9个候选框
            #为什么乘以grid_size 不明白
            xy = (xy * grid_size).view(fm_w, fm_h, 1, 2).expand(fm_w, fm_h, 9, 2)
            wh = self.anchor_edges[i].view(1, 1, 9, 2).expand(fm_w, fm_h, 9, 2)
            box = torch.cat([xy, wh], 3)  # [x, y, w, h]

            boxes.append(box.view(-1, 4))
            #五个级别按行排序
        return torch.cat(boxes, 0)

    def encode(self, boxes, labels, input_size):
        if isinstance(input_size, int):
            input_size = torch.Tensor([input_size, input_size])
        else:
            input_size = torch.Tensor(input_size)

        anchor_boxes = self.get_anchor_boxes(input_size)
        boxes = change_box_order(boxes, 'xyxy2xywh')
        boxes = boxes.float()
        ious = box_iou(anchor_boxes, boxes, order='xywh')
        #ious :每个候选框与多个目标框的ious 行：个数 列：分数
        #max_ids 与候选框最相近的目标框索引号，max_ious 为与最相近目标框的候选框的ious分数
        max_ious, max_ids = ious.max(1)
        #选出每个候选框对应的目标框
        boxes = boxes[max_ids]

        loc_xy = (boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:]
        loc_wh = torch.log(boxes[:, 2:] / anchor_boxes[:, 2:])
        loc_targets = torch.cat([loc_xy, loc_wh], 1)

        cls_targets = 1 + labels[max_ids]

        cls_targets[max_ious < 0.1] = 0
        cls_targets[(max_ious >= 0.1) & (max_ious < 0.3)] = -1
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, input_size):
        CLS_THRESH = 0.05
        NMS_THRESH = 0.3

        if isinstance(input_size, int):
            input_size = torch.Tensor([input_size, input_size])
        else:
            input_size = torch.Tensor(input_size)

        anchor_boxes = self.get_anchor_boxes(input_size)
        loc_xy = loc_preds[:, :2]
        loc_wh = loc_preds[:, 2:]
        xy = loc_xy * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
        wh = loc_wh.exp() * anchor_boxes[:, 2:]
        boxes = torch.cat([xy, wh], 1)
        boxes = change_box_order(boxes, 'xywh2xyxy')

        score, labels = cls_preds.max(1)
        ids = (score > CLS_THRESH) & (labels >0)
        ids = ids.nonzero().squeeze()
        keep = box_nms(boxes[ids], score[ids], threshold=NMS_THRESH)
        return boxes[ids][keep], labels[ids][keep]
