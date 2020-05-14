"""
Copyright (C) 2019 Scale Lab.  All rights reserved.
Licensed under the NTU license.
"""
from torch.utils.data.dataset import Dataset
import glob
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize,Grayscale
import numpy as np
import torch
import pickle
import os
import cv2

class My3DDataset(Dataset):
    def __init__(self,opts,is_Train=True):
        self.path = opts['data_root']
        self.shadow_root = opts['shadow_root']
        self.light_path = opts['light_root']
        self.scene_num = opts['scene_num'] #scale =9
        self.subject_index_num=opts['subject_index_num'] #scale=7


        self.is_use_dynamic = True

        self.is_Train = is_Train


        self.train_list,self.test_list,self.train_subjects,self.train_scenes,self.test_Yb_paths,self.test_Xb_paths \
            = self.split(opts['split_files_path'])
        self.size_train_subjects = len(self.train_subjects)
        self.size_train_scenes = len(self.train_scenes)
        if(self.is_Train):
            self.size = len(self.train_list)
        else:
            self.size = len(self.test_list)
        #transforms = []
        transforms = [Resize((opts['crop_image_height'], opts['crop_image_width']), Image.BICUBIC)]
        transforms.append(ToTensor())
        #transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)

        mask_transforms = [Resize((opts['crop_image_height'], opts['crop_image_width']), Image.BICUBIC)]
        mask_transforms.append(ToTensor())
        self.mask_transforms = Compose(mask_transforms)

        label_transforms = [Resize((16, 32), Image.BICUBIC)]
        label_transforms.append(ToTensor())
        self.label_transforms = Compose(label_transforms)


        self.generateMask()
        self.load_real()

        self.load_train_plus(opts)

        with open('data/subject_names.txt') as f:
            fr = f.readlines()
        self.subject_full_list = [x.strip() for x in fr]

        self.num_train = len(self.train_list)

        self.scale=2





    def split(self,split_file_path):
        if os.path.isfile(split_file_path):
            m = pickle.load(open(split_file_path, 'rb'))
            train_list =m['train_list']
            test_list = m['test_list']
            train_subjects=m['train_subject']
            train_scenes=m['train_scenes']
            test_Yb_paths=m['test_Yb_paths']
            test_Xb_paths=m['test_Xb_paths']
            return train_list,test_list,train_subjects,train_scenes,test_Yb_paths,test_Xb_paths


    def generateTestDynamic(self,index):
        Xa_path, Xb_path, Ya_path, Yb_path, Xa_prev_path, Xa_next_path, Xb_prev_path, Xb_next_path \
            = self.generateFour(self.test_list[index])
        Xa_out, Xa_mask = self.GetOne(Xa_path)
        Xa_prev_out, _ = self.GetOne(Xa_prev_path)
        Xa_next_out, _ = self.GetOne(Xa_next_path)
        return Xa_out,Xa_prev_out,Xa_next_out,Xa_mask


    def __getitem__(self, index):
        if(self.is_Train):
            Xa_path,light_a_path,Xb_path,light_b_path,r_Xb_path,diff,pattern = self.generateFour(self.train_list[index])
        else:
            Xa_path, light_a_path, Xb_path, light_b_path, r_Xb_path, diff, pattern = self.generateFour(
                self.train_test[index])

        Xa_out = self.transforms(Image.open(Xa_path).convert('RGB'))

        Xb_out = self.transforms(Image.open(Xb_path).convert('RGB'))

        mask = self.mask_dir[pattern]



        light_a_label = self.label_transforms(Image.open(light_a_path).convert('RGB'))
        light_b_label = self.label_transforms(Image.open(light_b_path).convert('RGB'))
        # rotation
        r_Xb_out = self.transforms(Image.open(r_Xb_path).convert('RGB'))

        return Xa_out,Xb_out,light_a_label,light_b_label,r_Xb_out,mask,diff

    # Xa, light_a, Xb, light_b,r_Xb
    def generateFour(self,Xa_path):
        tmp = Xa_path.split('/')
        if (tmp[self.shadow_plus_num] == 'shadow+'):
            X_pattern = tmp[-1].replace('.jpg', '')
            _subject_name = tmp[self.shadow_plus_subject_index_num]
            a_scene = tmp[self.shadow_plus_scene_num]
            a_scene_angle = 1
            # random
            b_scene = self.train_plus_scene[np.random.randint(0, self.size_train_plus_scene)]
            Xb_path = self.shadow_root + '../shadow+/%s/data/%s/%s.jpg' % (
                _subject_name, b_scene, X_pattern)
            pattern = X_pattern
            b_scene_angle = 1

            diff = 0
            rotated_angle = 1
            r_Xb_path = Xb_path
        else:
            a_scene = tmp[self.scene_num]
            a_scene_angle = tmp[-1].split('.')[1]
            X_subject_name = tmp[self.subject_index_num]

            # Xb(random)
            b_scene = self.train_scenes[np.random.randint(0, self.size_train_scenes)]
            b_scene_angle = '{:02}'.format(np.random.randint(1, 13))
            Xb_path = self.path + X_subject_name + '/data/' + b_scene + '/' + X_subject_name + '.' + b_scene_angle + '.jpg'

            pattern = X_subject_name

            # Rotate
            rotated_angle = '{:02}'.format(np.random.randint(1, 13))

            # light path
            r_Xb_path = self.path + X_subject_name + '/data/' + b_scene + '/' + X_subject_name + '.' + rotated_angle + '.jpg'
            diff = int(rotated_angle) - int(b_scene_angle)

        # light

        light_b_path = self.light_path + '%s/resize_%d.png' % (b_scene, int(b_scene_angle))
        light_a_path = self.light_path + '%s/resize_%d.png' % (a_scene, int(a_scene_angle))

        return Xa_path,light_a_path,Xb_path,light_b_path,r_Xb_path,diff,pattern



    def load_real(self):
        f = open('data/real_image.txt')
        fl = f.readlines()
        real_files =[]
        for i in range(len(fl)):
            real_files.append(fl[i].split(',')[0])
        self.real_files=real_files

    def getReal(self,index):
        file_path = self.path+'../Real_Image/'+self.real_files[index]
        beauty = self.transforms(Image.open(file_path+'.jpg').convert('RGB'))
        mask_ = self.mask_transforms(Image.open(file_path+'.tif').convert('RGB'))
        return beauty, mask_
        mask_path = self.path+'..'+'/Mask/*/data/albedo/*.png'
        dirs = glob.glob(mask_path)
        mask_dir={}
        for i in range(len(dirs)):
            pattern = dirs[i].split('/')[-1].split('.')[0]
            img = self.mask_transforms(Image.open(dirs[i]).convert('RGB'))
            mask_dir[pattern]=img
        self.mask_dir=mask_dir


    def load_train_plus(self,opts):
        #depth
        tmp = np.loadtxt('data/train_plus_scene.txt').astype(int)
        train_plus_scene=[]
        for i in tmp:
            train_plus_scene.append('Scene'+str(i))
        self.train_plus_scene = train_plus_scene
        self.size_train_plus_scene = len(self.train_plus_scene)

        #3D

        #mask
        mask_path = self.shadow_root+'../shadow+/mask/*/data/albedo/*.png'
        dirs = glob.glob(mask_path)
        mask_dir={}
        for i in range(len(dirs)):
            pattern = dirs[i].split('/')[-1].replace('.png','')
            img = self.mask_transforms(Image.open(dirs[i]).convert('RGB'))
            mask_dir[pattern]=img
        self.mask_dir.update(mask_dir)


        self.shadow_plus_num = opts['shadow_plus_num']
        self.shadow_plus_subject_index_num=opts['shadow_plus_subject_index_num']
        self.shadow_plus_scene_num=opts['shadow_plus_scene_num']

    def generateMask(self):
        mask_path = self.path+'..'+'/Mask/*/data/albedo/*.png'
        dirs = glob.glob(mask_path)
        mask_dir={}
        for i in range(len(dirs)):
            pattern = dirs[i].split('/')[-1].split('.')[0]
            img = self.mask_transforms(Image.open(dirs[i]).convert('RGB'))
            mask_dir[pattern]=img
        self.mask_dir=mask_dir



    def __len__(self):
        return self.size

    def getReal2(self,index):
        transforms = [Resize((256*self.scale, 256*self.scale), Image.BICUBIC)]
        transforms.append(ToTensor())
        #transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        beatuy_transforms = Compose(transforms)

        mask_transforms = [Resize((256*self.scale, 256*self.scale), Image.BICUBIC)]
        mask_transforms.append(ToTensor())
        mask_transforms = Compose(mask_transforms)


        file_path = self.path+'../Real_Image/'+self.real_files[index]
        beauty = beatuy_transforms(Image.open(file_path+'.jpg').convert('RGB'))
        mask_ = mask_transforms(Image.open(file_path+'.tif').convert('RGB'))
        return beauty, mask_