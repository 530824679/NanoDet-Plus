
import os
import cv2
import copy
import time
import logging
import torch
import numpy as np
from collections import defaultdict
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

def build_dataset(cfg, mode):
    dataset_cfg = copy.deepcopy(cfg)
    name = dataset_cfg.pop('name')
    if name == 'coco':
        return TXTDataset(mode=mode, **dataset_cfg)
    if name == 'xml_dataset':
        return XMLDataset(mode=mode, **dataset_cfg)
    else:
        raise NotImplementedError('Unknown dataset type!')

class BaseDataset(Dataset):
    def __init__(self, img_path, ann_path, input_size, pipeline, keep_ratio=True, use_instance_mask=False, use_seg_mask=False, use_keypoint=False, load_mosaic=False, mode='train'):
        self.img_path = img_path
        self.ann_path = ann_path
        self.input_size = input_size
        self.pipeline = Pipeline(pipeline, keep_ratio)
        self.keep_ratio = keep_ratio
        self.use_instance_mask = use_instance_mask
        self.use_seg_mask = use_seg_mask
        self.use_keypoint = use_keypoint
        self.load_mosaic = load_mosaic
        self.mode = mode

        self.data_info = self.get_data_info(ann_path)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if self.mode == 'val' or self.mode == 'test':
            return self.get_val_data(idx)
        else:
            while True:
                data = self.get_train_data(idx)
                if data is None:
                    idx = self.get_another_id()
                    continue
                return data

    @abstractmethod
    def get_data_info(self, ann_path):
        pass

    @abstractmethod
    def get_train_data(self, idx):
        pass

    @abstractmethod
    def get_val_data(self, idx):
        pass

    def get_another_id(self):
        return np.random.random_integers(0, len(self.data_info) - 1)

class TXTDataset(BaseDataset):

    def get_per_img_info(self, idx):
        img_info = self.data_info[idx]
        file_name = img_info['file_name']
        height = img_info['height']
        width = img_info['width']
        id = img_info['id']
        if not isinstance(id, int):
            raise TypeError('Image id must be int.')
        info = {'file_name': file_name,
                'height': height,
                'width': width,
                'id': id}
        return info

    def get_img_annotation(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco_api.getAnnIds([img_id])
        anns = self.coco_api.loadAnns(ann_ids)
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        if self.use_instance_mask:
            gt_masks = []
        if self.use_keypoint:
            gt_keypoints = []
        for ann in anns:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                if self.use_instance_mask:
                    gt_masks.append(self.coco_api.annToMask(ann))
                if self.use_keypoint:
                    gt_keypoints.append(ann['keypoints'])
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        annotation = dict(bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)
        if self.use_instance_mask:
            annotation['masks'] = gt_masks
        if self.use_keypoint:
            if gt_keypoints:
                annotation['keypoints'] = np.array(gt_keypoints, dtype=np.float32)
            else:
                annotation['keypoints'] = np.zeros((0, 51), dtype=np.float32)
        return annotation

    def get_train_data(self, idx):
        img_info = self.get_per_img_info(idx)
        file_name = img_info['file_name']
        image_path = os.path.join(self.img_path, file_name)
        img = cv2.imread(image_path)
        if img is None:
            print('image {} read failed.'.format(image_path))
            raise FileNotFoundError('Cant load image! Please check image path!')
        ann = self.get_img_annotation(idx)
        meta = dict(img=img,
                    img_info=img_info,
                    gt_bboxes=ann['bboxes'],
                    gt_labels=ann['labels'])
        if self.use_instance_mask:
            meta['gt_masks'] = ann['masks']
        if self.use_keypoint:
            meta['gt_keypoints'] = ann['keypoints']

        meta = self.pipeline(meta, self.input_size)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1))
        return meta

    def get_val_data(self, idx):
        # TODO: support TTA
        return self.get_train_data(idx)

class XMLDataset(BaseDataset):
    def __init__(self, class_names, **kwargs):
        self.class_names = class_names
        super(XMLDataset, self).__init__(**kwargs)

    def get_file_list(self, path, type='.xml'):
        file_names = []
        for root, sub, filelist in os.walk(path):
            for filename in filelist:
                file_path = os.path.join(root, filename)
                ext = os.path.splitext(file_path)[1]
                if ext == type:
                    file_names.append(filename)

        return file_names

    def xml_to_txt(self, ann_path):
        logging.info('loading annotations into memory...')
        tic = time.time()
        ann_file_names = self.get_file_list(ann_path, type='.xml')
        logging.info("Found {} annotation files.".format(len(ann_file_names)))
        image_info = []
        categories = []
        annotations = []
        for idx, supercat in enumerate(self.class_names):
            categories.append({'supercategory': supercat, 'id': idx + 1, 'name': supercat})

        ann_id = 1
        for idx, xml_name in enumerate(ann_file_names):
            tree = ET.parse(os.path.join(ann_path, xml_name))
            root = tree.getroot()
            file_name = root.find('filename').text
            width = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)
            info = {'file_name': file_name,
                    'height': height,
                    'width': width,
                    'id': idx + 1}
            image_info.append(info)
            for _object in root.findall('object'):
                category = _object.find('name').text
                if category not in self.class_names:
                    logging.warning("WARNING! {} is not in class_names! Pass this box annotation.".format(category))
                    continue
                for cat in categories:
                    if category == cat['name']:
                        cat_id = cat['id']
                xmin = int(_object.find('bndbox').find('xmin').text)
                ymin = int(_object.find('bndbox').find('ymin').text)
                xmax = int(_object.find('bndbox').find('xmax').text)
                ymax = int(_object.find('bndbox').find('ymax').text)
                w = xmax - xmin
                h = ymax - ymin
                if w < 0 or h < 0:
                    logging.warning("WARNING! Find error data in file {}! Box w and h should > 0. Pass this box annotation.".format(xml_name))
                    continue
                box_txt = [max(xmin, 0), max(ymin, 0), min(w, width), min(h, height)]
                ann = {'image_id': idx + 1,
                       'bbox': box_txt,
                       'category_id': cat_id,
                       'iscrowd': 0,
                       'id': ann_id,
                       'area': box_txt[2] * box_txt[3]
                       }
                annotations.append(ann)
                ann_id += 1

        dict_txt = {'images': image_info,
                     'categories': categories,
                     'annotations': annotations}
        logging.info('Load {} xml files and {} boxes'.format(len(image_info), len(annotations)))
        logging.info('Done (t={:0.2f}s)'.format(time.time() - tic))
        return dict_txt