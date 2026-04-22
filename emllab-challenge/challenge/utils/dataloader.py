import torch
import torch.utils.data
from torchinfo import summary
import os, json, os.path

import torchvision
from torchvision import transforms as tf

import albumentations as A

from challenge.utils import augmentation

VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
    )


def class_to_num(class_str):
    for idx, string in enumerate(VOC_CLASSES):
        if string == class_str: return idx

def num_to_class(number):
    for idx, string in enumerate(VOC_CLASSES):
        if idx == number: return string
    return 'none'

class VOCTransform:
    def __init__(self, train=True, only_person=False, more_augmentation=False):
        self.only_person = only_person
        self.train = train

        self.more_augmentation = more_augmentation

        if train:
            self.color_jitter = tf.RandomApply([tf.ColorJitter(0.2, 0.2, 0.2, 0.2)])

        if more_augmentation:
            self.transforms = augmentation.get_voc_transforms()


    def __call__(self,image, target):
        target = target['annotation']['object']

        if self.more_augmentation:
            image, target = augmentation.apply_voc_transforms(image, target, self.transforms)

        num_bboxes = 10
        width, height = 320, 320

        img_width, img_height = image.size

        scale = min(width/ img_width, height/img_height)
        new_width, new_height = int(img_width * scale), int( img_height * scale)

        diff_width, diff_height = width - new_width, height - new_height
        image = tf.functional.resize(image, size=(new_height, new_width))
        image = tf.functional.pad(image, padding = (diff_width//2,
                                                            diff_height//2,
                                                            diff_width//2 + diff_width % 2,
                                                            diff_height//2 + diff_height % 2))

        target_vectors = []
        for item in target:
            x0 = int(item['bndbox']['xmin'])*scale + diff_width//2
            w = (int(item['bndbox']['xmax']) - int(item['bndbox']['xmin']))* scale
            y0 = int(item['bndbox']['ymin'])*scale + diff_height//2
            h = (int(item['bndbox']['ymax']) - int(item['bndbox']['ymin'])) * scale

            target_vector = [(x0 + w/2) / width,
                            (y0 + h/2) / height,
                            w/width,
                            h/height,
                            1.0,
                            class_to_num(item['name'])]

            if self.only_person:
                if target_vector[5] == class_to_num("person"):
                    target_vector[5] = 0.0
                    target_vectors.append(target_vector)
            else:
                target_vectors.append(target_vector)

        target_vectors = list(sorted(target_vectors, key=lambda x: x[2]*x[3]))
        target_vectors = torch.tensor(target_vectors)
        if target_vectors.shape[0] < num_bboxes:
            zeros = torch.zeros((num_bboxes - target_vectors.shape[0], 6))
            zeros[:, -1] = -1
            target_vectors = torch.cat([target_vectors, zeros], 0)
        elif target_vectors.shape[0] > num_bboxes:
            target_vectors = target_vectors[:num_bboxes]

        if self.train:
            return self.color_jitter(tf.functional.to_tensor(image)), target_vectors,
        else:
            return tf.functional.to_tensor(image), target_vectors


def VOCDataset(data_dir: str, train: bool = True, only_person: bool = False, more_augmentation: bool = False) -> torchvision.datasets.VisionDataset:
    image_set = 'train' if train else 'val'

    download = not os.path.exists(os.path.join(data_dir, "VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg"))

    transform = VOCTransform(train=train, only_person=only_person, more_augmentation=more_augmentation)

    dataset = torchvision.datasets.VOCDetection(data_dir, year="2012", image_set=image_set, download=download, transforms=transform)

    if only_person:
        with open(os.path.join(data_dir, "person_indices_voc.json"), "r") as fd: indices = list(json.load(fd)[image_set])
        dataset = torch.utils.data.Subset(dataset, indices)
    
    return dataset


def VOCDataLoader(data_dir: str, train=True, batch_size=32, shuffle=False):
    dataset = VOCDataset(data_dir=data_dir, train=train)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
def VOCDataLoaderPerson(data_dir: str, train=True, batch_size=32, shuffle=False, more_augmentation=False):
    dataset = VOCDataset(data_dir=data_dir, train=train, only_person=True, more_augmentation=more_augmentation)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)




# https://www.kaggle.com/code/malikachhibber/coco-names
COCO_PERSON_CLASS_NUMBER = 1

class COCOTransform:
    def __init__(self, train=True, only_person=False, more_augmentation=False):
        self.only_person = only_person
        self.train = train

        self.more_augmentation = more_augmentation

        if train:
            self.color_jitter = tf.RandomApply([tf.ColorJitter(0.2, 0.2, 0.2, 0.2)])

        if more_augmentation:
            self.transforms = augmentation.get_coco_transforms()


    def __call__(self, image, target):
        if self.more_augmentation:
            image, target = augmentation.apply_coco_transforms(image, target, self.transforms)

        num_bboxes = 10
        width, height = 320, 320

        img_width, img_height = image.size

        scale = min(width/ img_width, height/img_height)
        new_width, new_height = int(img_width * scale), int( img_height * scale)

        diff_width, diff_height = width - new_width, height - new_height
        image = tf.functional.resize(image, size=(new_height, new_width))
        image = tf.functional.pad(image, padding = (diff_width//2,
                                                            diff_height//2,
                                                            diff_width//2 + diff_width % 2,
                                                            diff_height//2 + diff_height % 2))

        target_vectors = []
        for item in target:
            x0 = int(item['bbox'][0]) * scale + diff_width//2
            w = int(item['bbox'][2]) * scale
            y0 = int(item['bbox'][1]) * scale + diff_height//2
            h = int(item['bbox'][3]) * scale

            # skip small bounding boxes and some with invalid bbox labels
            if w * h < 100:
                continue

            target_vector = [(x0 + w/2) / width,
                            (y0 + h/2) / height,
                            w/width,
                            h/height,
                            1.0,
                            item['category_id']]

            if self.only_person:
                if target_vector[5] == COCO_PERSON_CLASS_NUMBER:
                    target_vector[5] = 0.0
                    target_vectors.append(target_vector)
            else:
                target_vectors.append(target_vector)

        target_vectors = list(sorted(target_vectors, key=lambda x: x[2]*x[3]))
        target_vectors = torch.tensor(target_vectors)
        if target_vectors.shape[0] < num_bboxes:
            zeros = torch.zeros((num_bboxes - target_vectors.shape[0], 6))
            zeros[:, -1] = -1
            target_vectors = torch.cat([target_vectors, zeros], 0)
        elif target_vectors.shape[0] > num_bboxes:
            target_vectors = target_vectors[:num_bboxes]

        if self.train:
            return self.color_jitter(tf.functional.to_tensor(image)), target_vectors,
        else:
            return tf.functional.to_tensor(image), target_vectors

def COCODataset(data_dir: str, train: bool = True, only_person: bool = False, more_augmentation: bool = False, do_transform: bool = True) -> torchvision.datasets.VisionDataset:
    # Using the COCO Dataset throws errors due to truncated images, this is a workaround -> corrupted file?
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    image_set = 'train2017' if train else 'val2017'
    root = os.path.join(data_dir, f'COCO/{image_set}')
    ann_file = os.path.join(data_dir, f'COCO/annotations/instances_{image_set}.json')

    transform = COCOTransform(train=train, only_person=only_person, more_augmentation=more_augmentation) if do_transform else None
    dataset = torchvision.datasets.CocoDetection(root=root, annFile=ann_file, transforms=transform)

    if only_person:
        with open(os.path.join(data_dir, 'person_indices_coco.json'), 'r') as fd: indices = list(json.load(fd)[image_set])
        dataset = torch.utils.data.Subset(dataset, indices)

    return dataset


def COCODataLoader(data_dir: str, train=True, batch_size=32, shuffle=False):
    dataset = COCODataset(data_dir=data_dir, train=train)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
def COCODataLoaderPerson(data_dir: str, train=True, batch_size=32, shuffle=False, more_augmentation=False):
    dataset = COCODataset(data_dir=data_dir, train=train, only_person=True, more_augmentation=more_augmentation)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def VOC_plus_COCO_DataLoaderPerson(data_dir: str, train=True, batch_size=32, shuffle=False, more_augmentation=True):
    voc = VOCDataset(data_dir, train, only_person=True, more_augmentation=more_augmentation)
    coco = COCODataset(data_dir, train, only_person=True, more_augmentation=more_augmentation)
    combined = torch.utils.data.ConcatDataset([voc, coco])
    return torch.utils.data.DataLoader(combined, batch_size=batch_size, shuffle=shuffle)