import os
import torch
import cv2
import torch.utils.data as data
from pycocotools import COCO

class CocoDataset(data.Dataset):
    def __init__(self, root, annotation, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)

        target = coco.loadAnns(ann_ids)
        target = torch.unsqueeze(torch.Tensor(target[0]['bbox']), -1)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(self.root, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    

    def __len__(self):
        return len(self.ids)

