import os
import numpy as np
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
import cv2
from collections import defaultdict
from typing import Any, Dict, List, Tuple

class PascalDataset(Dataset):
    def __init__(self, root: str, img_set: str = 'train', transform: Any = None) -> None:
        super(PascalDataset, self).__init__()
        self.root: str = root
        self.img_set: str = img_set
        self.transform: Any = transform

        voc_root: str = os.path.join(self.root, "VOCdevkit", "VOC2007")
        img_dir: str = os.path.join(voc_root, "JPEGImages")
        annotation_dir: str = os.path.join(voc_root, "Annotations")
        img_set_file: str = os.path.join(voc_root, "ImageSets", "Main", f"{img_set}.txt")
        
        with open(img_set_file) as f:
            self.ids: List[str] = [line.strip() for line in f]

        self.img_annotation_paths: List[Tuple[str, str]] = [(os.path.join(img_dir, f"{img_id}.jpg"), os.path.join(annotation_dir, f"{img_id}.xml")) for img_id in self.ids]
        self.classes: List[str] = self._get_classes()
        self.class_to_idx: Dict[str, int] = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def _get_classes(self) -> List[str]:
        classes: set = set()
        for _, annotation_path in self.img_annotation_paths:
            tree = ET.parse(annotation_path)
            for obj in tree.findall('object'):
                classes.add(obj.find('name').text)
        return sorted(list(classes))
    
    def _parse_voc_xml(self, node: ET.Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children: List[ET.Element] = list(node)
        if children:
            def_dic: defaultdict = defaultdict(list)
            for dc in map(self._parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag: {ind: v[0] if len(v) == 1 else v
                           for ind, v in def_dic.items()}}
        if node.text:
            text: str = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path, annotation_path = self.img_annotation_paths[idx]

        img: np.ndarray = cv2.imread(img_path)
        img: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mean: np.ndarray = np.array([0.485, 0.456, 0.406])
        std: np.ndarray = np.array([0.229, 0.224, 0.225])
        img: np.ndarray = (img / 255.0 - mean) / std
        assert not np.any(np.isnan(img))

        img: torch.Tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

        tree: ET.ElementTree = ET.parse(annotation_path)
        target: Dict[str, Any] = self._parse_voc_xml(tree.getroot())['annotation']

        boxes: List[List[float]] = []
        labels: List[int] = []

        # Ensure 'object' key exists in target
        if 'object' in target:
            objects: Any = target['object']
            if not isinstance(objects, list):
                objects = [objects]
            for obj in objects:
                bbox: Dict[str, Any] = obj['bndbox']
                x_min: float = float(bbox['xmin'])
                y_min: float = float(bbox['ymin'])
                x_max: float = float(bbox['xmax'])
                y_max: float = float(bbox['ymax'])
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(self.class_to_idx[obj['name']])
        else:
            print(f"No objects found in {annotation_path}")

        boxes: torch.Tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels: torch.Tensor = torch.as_tensor(labels, dtype=torch.int64)

        target: Dict[str, torch.Tensor] = {'boxes': boxes, 'labels': labels}

        if self.transform:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.ids)
