import torch
import torchvision.ops as ops

def decode_boxes(regression, anchors, variances=[0.1, 0.2]):
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx = regression[:, 0] * variances[0]
    dy = regression[:, 1] * variances[0]
    dw = regression[:, 2] * variances[1]
    dh = regression[:, 3] * variances[1]

    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.zeros_like(regression)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def nms(boxes, scores, threshold):
    nms = ops.nms(boxes, scores, threshold)
    return nms

def postprocess(cls_outputs, reg_outputs, anchors, score_threshold=0.05, nms_threshold=0.5, max_detections=100):
    batch_size = cls_outputs[0].shape[0]
    num_classes = cls_outputs[0].shape[-1]
    
    all_detections = []

    for i in range(batch_size):
        cls_scores = [cls_output[i] for cls_output in cls_outputs]
        cls_scores = torch.cat([score.view(-1, num_classes) for score in cls_scores], dim=0)

        reg_scores = [reg_output[i] for reg_output in reg_outputs]
        reg_scores = torch.cat([score.view(-1, 4) for score in reg_scores], dim=0)

        anchors_batch = torch.cat([a.view(-1, 4) for a in anchors], dim=0)

        cls_scores = cls_scores.sigmoid()
        max_scores, labels = torch.max(cls_scores, dim=1)
        
        keep = max_scores > score_threshold
        max_scores = max_scores[keep]
        labels = labels[keep]
        reg_scores = reg_scores[keep]
        anchors_batch = anchors_batch[keep]

        decoded_boxes = decode_boxes(reg_scores, anchors_batch)

        keep = nms(decoded_boxes, max_scores, nms_threshold)

        if keep.size(0) > max_detections:
            keep = keep[:max_detections]

        final_boxes = decoded_boxes[keep]
        final_scores = max_scores[keep]
        final_labels = labels[keep]

        detections = []
        for j in range(final_boxes.size(0)):
            detections.append({
                'bbox': final_boxes[j].cpu().numpy(),
                'score': final_scores[j].cpu().numpy(),
                'class': final_labels[j].cpu().numpy()
            })

        all_detections.append(detections)
    
    return all_detections

def perform_inference(model, dataloader, device, score_threshold=0.05, nms_threshold=0.5):
    """
    Perform inference on the model.
    """
    model.to(device)
    model.eval()
    detections = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            cls_outputs, reg_outputs = model(images)

            anchors = generate_anchors_for_image(images.shape[2:], cls_outputs)
            
            batch_detections = postprocess(cls_outputs, reg_outputs, anchors, score_threshold, nms_threshold)
            detections.extend(batch_detections)

    return detections

def generate_anchors_for_image(image_shape, cls_outputs):
    return [torch.zeros(cls_output.size(0), 4) for cls_output in cls_outputs]
