import torch 
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_anchors(base_size=16, ratios=None, scales=None):
    if ratios is None:
        ratios = torch.tensor([0.5, 1, 2], device=device)
    if scales is None:
        scales = torch.tensor([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], device=device)
    num_anchors = len(ratios) * len(scales)
    anchors = torch.zeros((num_anchors, 4), device=device)
    anchors[:, 2:] = base_size * scales.repeat(2, len(ratios)).t()
    areas = anchors[:, 2] * anchors[:, 3]
    anchors[:, 2] = torch.sqrt(areas / ratios.repeat(len(scales)))
    anchors[:, 3] = anchors[:, 2] * ratios.repeat(len(scales))
    anchors[:, 0::2] -= anchors[:, 2].unsqueeze(1) * 0.5
    anchors[:, 1::2] -= anchors[:, 3].unsqueeze(1) * 0.5

    return anchors

def compute_shape(image_shape, pyramid_levels):
    image_shape = torch.tensor(image_shape[:2], device=device)
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes

def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None,
):
    image_shapes = compute_shape(image_shape, pyramid_levels)

    all_anchors = torch.zeros((0, 4), device=device)
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors = torch.cat([all_anchors, shifted_anchors])

    return all_anchors

def shift(shape, stride, anchors):
    shift_x = (torch.arange(0, shape[1], device=device) + 0.5) * stride
    shift_y = (torch.arange(0, shape[0], device=device) + 0.5) * stride

    shift_x, shift_y = torch.meshgrid(shift_x, shift_y, indexing='ij')

    shifts = torch.stack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    ), dim=1)

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.view((1, A, 4)) + shifts.view((1, K, 4)).permute((1, 0, 2)))
    all_anchors = all_anchors.view((K * A, 4))

    return all_anchors

class Anchors(nn.Module):
    def __init__(self, pyramid_level=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()

        if pyramid_level is None:
            self.pyramid_levels = [3,4,5,6,7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x+2) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = torch.tensor([0.5, 1, 2], device=device)
        if scales is None:
            self.scales = torch.tensor([2**0, 2**(1.0/3.0), 2**(2.0/3.0)], device=device)

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = torch.tensor(image_shape, device=device)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        all_anchors = torch.zeros((0, 4), dtype=torch.float32, device=device)

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = torch.cat([all_anchors, shifted_anchors])

        return all_anchors.unsqueeze(0)

class BoundingBoxTransform(nn.Module):
    def __init__(self, mean=None, std=None):
        super(BoundingBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.tensor([0, 0, 0, 0], dtype=torch.float32, device=device)
        else:
            self.mean = mean
        if std is None:
            self.std = torch.tensor([0.1, 0.1, 0.2, 0.2], dtype=torch.float32, device=device)
        else:
            self.std = std

    def forward(self, boxes, deltas):
        widths  = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x   = boxes[:, :, 0] + 0.5 * widths
        ctr_y   = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes

class ClipBoxes(nn.Module):
    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape
        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)
        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
      
        return boxes