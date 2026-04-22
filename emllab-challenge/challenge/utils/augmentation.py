import PIL.Image
import numpy as np
import cv2

import copy

import albumentations as A

def get_voc_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT),
    ], bbox_params=A.BboxParams(format='pascal_voc'))


def apply_voc_transforms(image: PIL.Image, target: list[dict], transforms: A.Compose) -> (PIL.Image, list[dict]):
    target = copy.deepcopy(target) # copy to not overwrite the original
    boxes = []
    for t in target:
        bbx = t['bndbox']
        boxes.append([int(bbx['xmin']), int(bbx['ymin']), int(bbx['xmax']), int(bbx['ymax']), 'fake_label'])
    
    image_np = np.array(image)

    transformed = transforms(image=image_np, bboxes=boxes)
    image_np, boxes = transformed['image'], transformed['bboxes']

    image = PIL.Image.fromarray(image_np)

    for t, b in zip(target, boxes):
        t['bndbox']['xmin'] = b[0]
        t['bndbox']['ymin'] = b[1]
        t['bndbox']['xmax'] = b[2]
        t['bndbox']['ymax'] = b[3]
    return image, target



def get_coco_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT),
    ], bbox_params=A.BboxParams(format='coco'))


def apply_coco_transforms(image: PIL.Image, target: list[dict], transforms: A.Compose) -> (PIL.Image, list[dict]):
    target = copy.deepcopy(target) # copy to not overwrite the original
    boxes = []
    for t in target:
        bbx = t['bbox'] # copy to not overwrite the original (target was not deepcopied)

        # fix for invalid bounding boxes in COCO
        if bbx[2] == 0:
            bbx[2] = 0.001
            #print(f'fixed width of bbox: {bbx}')
        if bbx[3] == 0:
            bbx[3] = 0.001
            #print(f'fixed height of bbox: {bbx}')

        bbx.append('fake_label')
        boxes.append(bbx)

    image_np = np.array(image)

    transformed = transforms(image=image_np, bboxes=boxes)
    image_np, boxes = transformed['image'], transformed['bboxes']

    image = PIL.Image.fromarray(image_np)

    for t, b in zip(target, boxes):
        t['bbox'] = b[:4] # don't use fake label

    return image, target
