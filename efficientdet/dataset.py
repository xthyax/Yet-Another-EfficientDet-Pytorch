import os
import torch
import numpy as np
import imghdr
import glob
import json
import random
from torch.utils.data import Dataset, DataLoader
# from pycocotools.coco import COCO
import cv2


class DataGenerator(Dataset):
    def __init__(self, dataset_dir, classes, transform=None, augmentation=None):

        if isinstance(dataset_dir , list):
            self.dataset_dir = dataset_dir
        else:
            self.dataset_dir = [dataset_dir]

        self.transform = transform

        self.image_paths = self.load_image_path()

        self.load_classes(classes)

        self.augmentation = augmentation

    def load_image_path(self):
        # Get image name for image id
        list_imgs_path = []
        for path_data in self.dataset_dir:

            if "train" in path_data.lower().split("\\")[-1]:
                path_data = os.path.join(self.dataset_dir, "OriginImage")
            else:
                pass
            
            for image_path in glob.glob(os.path.join(path_data,"*.bmp")):
                # print(image)
                if imghdr.what(image_path):
                    # image_name = os.path.split(image)[1]
                    list_imgs_path.append(image_path)
                
        return list_imgs_path


    def load_classes(self, classes_list):

        self.classes = {}
        for c_index, c in enumerate(classes_list):
            self.classes[classes_list[c_index]] = len(self.classes)
        # print(self.classes)
        # also load the reverse (label -> name)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key
        # print(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img, augment_opt = self.load_image(idx)
        annot = self.load_annotations(idx, augment_opt)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_path = self.image_paths[image_index]
        image_name = os.path.split(image_path)[1]
        random_augment = self.augmentation and torch.randint(0, 2,(1,)).bool().item()

        if random_augment:
            path = os.path.join(self.dataset_dir[0],"TransformImage", random.choices(self.augmentation)+"_"+image_name)
        else:
            path = image_path
        # path = os.path.join(self.dataset_dir, image_info)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print(image_info)
        return img.astype(np.float32) / 255., random_augment

    def load_annotations(self, image_index, augment_opt):
        # get ground truth annotations
        # print(self.image_ids)
        image_path = self.image_paths[image_index]
        image_name = os.path.split(image_path)[1]
        try:
            if augment_opt:
                with open(os.path.join(self.dataset_dir[0],"TransformImage",image_name + ".json")) as f:
                    obj = json.load(f)
            else:
                with open(image_path + ".json") as f:
                    obj = json.load(f)
        except:
            pass
        
        annos = obj['regions']
        class_name = obj['classId']
        bboxes= []
        for annos_position in annos:

            px = annos[annos_position]["List_X"]
            py = annos[annos_position]["List_Y"]
        
            poly = np.stack((px, py), axis=1)
            maxxy = poly.max(axis=0)
            minxy = poly.min(axis=0)
            
            bboxes.append([minxy[0], minxy[1], maxxy[0], maxxy[1], class_name[int(annos_position)]])

        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(bboxes) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(bboxes):

            # some annotations have basically no width / height, skip them
            # if a['bbox'][2] < 1 or a['bbox'][3] < 1:
            #     continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = bboxes[idx][:4]
            annotation[0, 4] =  self.classes[bboxes[idx][4]]
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        # annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        # annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        np.random.seed(1)
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
