import os
import xml.etree.ElementTree as ET
import torch
import torch.utils.data.dataset as Dataset
import cv2
from collections import defaultdict

class PascalDataset(Dataset):
    def __init__(self, root, img_set='train', transform=None) -> None:
        super(PascalDataset, self).__init__()
        self.root = root
        self.img_set = img_set
        self.transform = transform

        voc_root = os.path.join(self.root, "VOC2007")
        img_dir = os.path.join(voc_root, "JPEGImages")
        annotation_dir = os.path.join(voc_root, "Annotations")
        img_set_file = os.path.join(voc_root, "ImageSets", "Main", f"{img_set}.txt")
        
        with open(img_set_file) as f:
            self.ids = [line.strip() for line in f]

        self.img_paths = [os.path.join(img_dir, f"{img_id}.jpg") for img_id in self.ids]
        self.annotation_paths = [os.path.join(annotation_dir, f"{img_id}.xml") for img_id in self.ids]
        self.classes = self._get_classes()
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
    
    def _get_classes(self) -> set:
        classes = set()
        for annotation_path in self.annotation_paths:
            tree = ET.parse(annotation_path)
            for obj in tree.findall('object'):
                classes.add(obj.find('name').text)
        return sorted(classes)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        annotation_path = self.annotation_paths[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tree = ET.parse(annotation_path)
        target = self._parse_voc_xml(tree.getroot())

        boxes = []
        labels = []

        for obj in target['annotation']['object']:
            bbox = obj['bndbox']
            x_min = float(bbox['xmin'])
            y_min = float(bbox['ymin'])
            x_max = float(bbox['xmax'])
            y_max = float(bbox['ymax'])
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.class_to_idx[obj['name']])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels}

        if self.transform:
            img = self.transform(img)
        
        return img, target
    
    def __len__(self):
        return len(self.ids)
    
    def parse_voc_xml(self, node) -> dict:
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {node.tag: {k: v[0] if len(v) == 1 else v for k, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict        