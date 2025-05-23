import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation
from torchvision import transforms
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop
    
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


class PromptTrainDataset(Dataset):
    def __init__(self, args):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur' : 5, 'delowlight' : 6.}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()
        
        normalize = Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
        self.transform_clip = transforms.Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            normalize
        ])

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()
        if 'deblur' in self.de_type:
            self._init_blurry_ids()
        if 'delowlight' in self.de_type:
            self._init_lowlight_ids()

        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        clean_paths = []
        degraded_paths = []
        with open(self.args.denoise15_dir, 'r') as file:
            for line in file:
                degraded_paths.append((line.split(',')[0].strip()).replace('.', './datasets', 1))
                clean_paths.append((line.split(',')[1].strip()).replace('.', './datasets', 1))
        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"degraded_id":degraded,"clean_id":clean,"de_type":0} for degraded, clean in zip(degraded_paths, clean_paths)]
            self.s15_ids = self.s15_ids * 3
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        self.num_clean = len(clean_paths)
        print("Total Denoise15 Ids : {}".format(self.num_clean))
        
        clean_paths = []
        degraded_paths = []
        with open(self.args.denoise25_dir, 'r') as file:
            for line in file:
                degraded_paths.append((line.split(',')[0].strip()).replace('.', './datasets', 1))
                clean_paths.append((line.split(',')[1].strip()).replace('.', './datasets', 1))
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"degraded_id":degraded,"clean_id":clean,"de_type":1} for degraded, clean in zip(degraded_paths, clean_paths)]
            self.s25_ids = self.s25_ids * 3
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        self.num_clean = len(clean_paths)
        print("Total Denoise25 Ids : {}".format(self.num_clean))
        
        clean_paths = []
        degraded_paths = []
        with open(self.args.denoise50_dir, 'r') as file:
            for line in file:
                degraded_paths.append((line.split(',')[0].strip()).replace('.', './datasets', 1))
                clean_paths.append((line.split(',')[1].strip()).replace('.', './datasets', 1))
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"degraded_id":degraded,"clean_id":clean,"de_type":2} for degraded, clean in zip(degraded_paths, clean_paths)]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0
        self.num_clean = len(clean_paths)
        print("Total Denoise50 Ids : {}".format(self.num_clean))

    def _init_hazy_ids(self):
        clean_paths = []
        degraded_paths = []
        with open(self.args.dehaze_dir, 'r') as file:
            for line in file:
                degraded_paths.append((line.split(',')[0].strip()).replace('.', './datasets', 1))
                clean_paths.append((line.split(',')[1].strip()).replace('.', './datasets', 1))
        self.hazy_ids = [{"degraded_id":degraded,"clean_id":clean,"de_type":4} for degraded, clean in zip(degraded_paths, clean_paths)]
        self.hazy_counter = 0
        
        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_rs_ids(self):
        clean_paths = []
        degraded_paths = []
        with open(self.args.derain_dir, 'r') as file:
            for line in file:
                degraded_paths.append((line.split(',')[0].strip()).replace('.', './datasets', 1))
                clean_paths.append((line.split(',')[1].strip()).replace('.', './datasets', 1))
        
        self.rs_ids = [{"degraded_id":degraded,"clean_id":clean,"de_type":3} for degraded, clean in zip(degraded_paths, clean_paths)]
        if self.args.is_addRainSets == False:
            self.rs_ids = self.rs_ids * 120

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))
        
    def _init_blurry_ids(self):
        clean_paths = []
        degraded_paths = []
        with open(self.args.deblur_dir, 'r') as file:
            for line in file:
                degraded_paths.append((line.split(',')[0].strip()).replace('.', './datasets', 1))
                clean_paths.append((line.split(',')[1].strip()).replace('.', './datasets', 1))
        
        self.blurry_ids = [{"degraded_id":degraded,"clean_id":clean,"de_type":5} for degraded, clean in zip(degraded_paths, clean_paths)]

        self.blurry_counter = 0
        self.num_blurry = len(self.blurry_ids)
        print("Total Blurry Ids : {}".format(self.num_blurry))
        
    def _init_lowlight_ids(self):
        clean_paths = []
        degraded_paths = []
        with open(self.args.delowlight_dir, 'r') as file:
            for line in file:
                degraded_paths.append((line.split(',')[0].strip()).replace('.', './datasets', 1))
                clean_paths.append((line.split(',')[1].strip()).replace('.', './datasets', 1))
        
        self.lowlight_ids = [{"degraded_id":degraded,"clean_id":clean,"de_type":6} for degraded, clean in zip(degraded_paths, clean_paths)]

        self.lowlight_counter = 0
        self.num_lowlight = len(self.lowlight_ids)
        print("Total lowlight Ids : {}".format(self.num_lowlight))
    

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
            self.sample_ids += self.s25_ids
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids+= self.rs_ids
        if "dehaze" in self.de_type:
            self.sample_ids+= self.hazy_ids
        if "deblur" in self.de_type:
            self.sample_ids+= self.blurry_ids
        if "delowlight" in self.de_type:
            self.sample_ids+= self.lowlight_ids
        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
            
        degrad_img = crop_img(np.array(Image.open(sample["degraded_id"]).convert('RGB')), base=16)
        clean_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
            
        clean_name = sample["degraded_id"].split("/")[-1].split('.')[0]
        
        degrad_patch, clean_patch = self._crop_patch(degrad_img, clean_img)
        
        degrad_patch, clean_patch = random_augmentation(degrad_patch, clean_patch)

        degrad_patch = self.toTensor(degrad_patch)
        clean_patch = self.toTensor(clean_patch)
        
        return [clean_name, de_id], degrad_patch, clean_patch, self.transform_clip(degrad_patch)

    def __len__(self):
        return len(self.sample_ids)


class TestDataset_forIR(Dataset):
    def __init__(self, path, type, sets_name):
        self.image_paths = []
        self.image_gt_paths = []
        self.type = type
        self.sets_name = sets_name
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
        return image, gt, clip_input, cur_type, os.path.basename(img_path), self.sets_name