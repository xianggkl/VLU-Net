import os
from torch.utils.data import Dataset, DataLoader
from utils.image_utils import random_augmentation, crop_img
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop
from torchvision import transforms
from PIL import Image
import numpy as np
from utils.val_utils import AverageMeter, compute_psnr_ssim

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

class RandomCropPair:
    def __init__(self, size):
        self.size = size
        self.resize = transforms.Resize(size)
    def __call__(self, img, gt):
        if img.shape[0] < self.size or img.shape[1] < self.size:
            img = self.resize(img)
            gt = self.resize(gt)
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.size, self.size))
        img = transforms.functional.crop(img, i, j, h, w)
        gt = transforms.functional.crop(gt, i, j, h, w)
        return img, gt
    
class TrainDataset_forIR(Dataset):
    def __init__(self, paths_list, opt):
        self.opt = opt
        self.crop = RandomCropPair(self.opt.patch_size)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        normalize = Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
        self.transform_clip = transforms.Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            normalize
        ])
        
        self.image_paths = []
        self.image_gt_paths = []
        self.types = []
        for index, path in enumerate(paths_list):
            with open(path, 'r') as file:
                img_paths = []
                gt_paths = []
                cur_type = []
                for line in file:
                    img_paths.append((line.split(',')[0].strip()).replace('.', './datasets', 1))
                    gt_paths.append((line.split(',')[1].strip()).replace('.', './datasets', 1))
                    cur_type.append(index)
                if index == 4:
                    img_paths = img_paths * 120
                    gt_paths = gt_paths * 120
                    cur_type = cur_type * 120
                self.image_paths.extend(img_paths)
                self.image_gt_paths.extend(gt_paths)
                self.types.extend(cur_type)
                print(len(img_paths))

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_gt_path = self.image_gt_paths[idx]
        cur_type = self.types[idx]
        
        image = self.transform(crop_img(np.array(Image.open(img_path).convert('RGB')), base=16))
        gt = self.transform(crop_img(np.array(Image.open(img_gt_path).convert('RGB')), base=16))
        
        image, gt = self.crop(image, gt)
        if self.opt.is_aug:
            image, gt = random_augmentation(image, gt)
        clip_input = self.transform_clip(image)
        
        return image, gt, clip_input, cur_type
    

class TestDataset_forIR(Dataset):
    def __init__(self, path, type):
        self.image_paths = []
        self.image_gt_paths = []
        self.type = type
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        normalize = Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
        self.transform_clip = transforms.Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            normalize
        ])
        with open(path, 'r') as file:
            for line in file:
                self.image_paths.append((line.split(',')[0].strip()).replace('.', './datasets', 1))
                self.image_gt_paths.append((line.split(',')[1].strip()).replace('.', './datasets', 1))
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_gt_path = self.image_gt_paths[idx]
        cur_type = self.type
        image = self.transform(crop_img(np.array(Image.open(img_path).convert('RGB')), base=16))
        gt = self.transform(crop_img(np.array(Image.open(img_gt_path).convert('RGB')), base=16))
        clip_input = self.transform_clip(image)
        return image, gt, clip_input, cur_type, os.path.basename(img_path)