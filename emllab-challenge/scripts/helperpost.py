import cv2
import numpy as np
from typing import List

"""
Calculate iou between each bbox in 'bboxes1' with each bbox in 'bboxes2'.

Args: 
    bboxes1: bounding box
    bboxes2: bounding box

Returns: 
    ious
"""
def npIou(bboxes1, bboxes2):
    px, py, pw, ph = np.array_split(bboxes1[...,:4].reshape(-1, 4), 4, axis=-1)
    lx, ly, lw, lh = np.array_split(bboxes2[...,:4].reshape(-1, 4), 4, axis=-1)
    px1, py1, px2, py2 = px - 0.5 * pw, py - 0.5 * ph, px + 0.5 * pw, py + 0.5 * ph
    lx1, ly1, lx2, ly2 = lx - 0.5 * lw, ly - 0.5 * lh, lx + 0.5 * lw, ly + 0.5 * lh
    zero = np.array(0.0, dtype=px1.dtype)
    
    dx = np.maximum(np.minimum(px2, lx2.T) - np.maximum(px1, lx1.T), zero)
    dy = np.maximum(np.minimum(py2, ly2.T) - np.maximum(py1, ly1.T), zero)
    intersections = dx * dy
    pa = (px2 - px1) * (py2 - py1) # area
    la = (lx2 - lx1) * (ly2 - ly1) # area
    unions = (pa + la.T) - intersections
    ious = (intersections/unions).reshape(*bboxes1.shape[:-1], *bboxes2.shape[:-1])

    return ious


"""
Filter bounding boxes.

Args: 
    output_ndarray: output of yolo
    threshold:      threshold of confidence

Returns: 
    filtered: List[np.ndarray]
"""
def npFilter_boxes(output_ndarray: np.ndarray, threshold) -> List[np.ndarray]:
    b, a, h, w, c = output_ndarray.shape
    x = output_ndarray.reshape(b, a * h * w, c)
    boxes = x[:, :, 0:4]
    confidence = x[:, :, 4]
    scores = np.max(x[:, :, 5:], axis=-1)
    idx = np.argmax(x[:, :, 5:], axis=-1)
    
    idx = np.float32(idx)
    scores = scores * confidence
    mask = scores > threshold
    
    filtered = []
    for c, s, i, m in zip(boxes, scores, idx, mask):
        if m.any():
            detected = np.concatenate((c[m, :], s[m, None], i[m, None]), -1)
        else:
            detected = np.zeros((0, 6), dtype=x.dtype)
        filtered.append(detected)
    return filtered


"""
Non maximum suppression.

Args: 
    filtered_ndarray: current frame or image to draw in
    threshold: threshold of nms

Returns: 
    result: List[np.ndarray] of detections
"""
def npNms(filtered_ndarray: List[np.ndarray], threshold: float) -> List[np.ndarray]:
    result = []
    for x in filtered_ndarray:
        # Sort coordinates by descending confidence
        order = np.argsort(x[:, 4], 0)
        order = order[::-1]
        x = x[order]
        ious = npIou(x,x) # get ious between each bbox in x

        # Filter based on iou
        keep = (ious > threshold).astype(np.int_)
        keep = np.triu(keep, 1)
        keep = np.sum(keep, 0, keepdims=True)
        keep = keep.T
        keep = np.broadcast_to(keep, x.shape) == 0

        result.append(x[keep].reshape(-1, 6))
    return result


"""
Display boxes of the detected object.

Args: 
    frame: current frame or image to draw in
    output: list of bounding boxes

Returns: frame
"""
def npDisplayBoxes(frame, output: List[np.ndarray]):
    # frame shape is (320,320,3)
    pad = 20
    frame = np.pad(frame, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
    img_shape = 320
        
    if output:
        bboxes = np.stack(output, axis=0)
        for i in range(bboxes.shape[1]):
            if bboxes[0,i,-1] >= 0:
                '''                
                Index meanings:
                0 - x center
                1 - y center
                2 - Width
                3 - Height
                4 - confidence
                5 - class idx
                '''                
                # top left corner
                start_point = (
                    int(bboxes[0,i,0]*img_shape - bboxes[0,i,2]*img_shape/2) - pad,
                    int(bboxes[0,i,1]*img_shape - bboxes[0,i,3]*img_shape/2) - pad
                )
                # bottom right corner
                end_point = (
                    int(bboxes[0,i,0]*img_shape + bboxes[0,i,2]*img_shape/2) + pad,
                    int(bboxes[0,i,1]*img_shape + bboxes[0,i,3]*img_shape/2) + pad
                )
                color = (255, 0, 0)  # BGR
                #print(num_to_class(int(bboxes[0,i,5])))
                #print((start_point, end_point))
                
                cv2.rectangle(frame, start_point, end_point, color, 2) 

    return frame



