import os
from PIL import Image
import numpy as np
from random import uniform
from utils import *
from tqdm import tqdm

pano_dir = '../sample_data'
fov_dir = '../fov_dataset'
modes = ['train', 'val', 'test']

split = ['albedo', 'shading', 'original', 'environment']

height, width = 240, 320
pxs = height * width

whole_albedo_list = sorted(os.listdir(os.path.join(pano_dir,split[0])))
whole_shading_list = sorted(os.listdir(os.path.join(pano_dir,split[1])))
whole_original_list = sorted(os.listdir(os.path.join(pano_dir,split[2])))
whole_environment_list = sorted(os.listdir(os.path.join(pano_dir,split[3])))

assert len(whole_albedo_list) == len(whole_shading_list) == len(whole_original_list) == len(whole_environment_list)

for mode in modes:
    print('='*20)
    print(f'Create {mode} dataset...')
    
    # train/val/test split
    if mode == 'train':
        start, end = 0, 69
    elif mode == 'val':
        start, end = 70, 79
    elif mode == 'test':
        start, end = 80, 99
    albedo_list = whole_albedo_list[start:end]
    shading_list = whole_shading_list[start:end]
    original_list = whole_original_list[start:end]
    environment_list = whole_environment_list[start:end]
    
    for i in tqdm(range(len(albedo_list))):
        a = np.asarray(Image.open(os.path.join(pano_dir, 'albedo', albedo_list[i])))
        s = np.asarray(Image.open(os.path.join(pano_dir, 'shading', shading_list[i])))
        o = np.asarray(Image.open(os.path.join(pano_dir, 'original', original_list[i])))
        e = np.asarray(Image.open(os.path.join(pano_dir, 'environment', environment_list[i])))

        equ_a = Equirectangular(a)
        equ_s = Equirectangular(s)
        equ_o = Equirectangular(o)
        equ_e = Equirectangular(e)

        count = 0
        while count < 20:
            # random parameters for cropping
            fov, horizon, azimuth = round(uniform(35, 68), 3), round(uniform(-20, 20), 3), round(uniform(-180,180))

            # check ratio of valid area 
            img_e = equ_e.GetPerspective(fov, horizon, azimuth, 240, 320)
            ratio = (img_e[:,:,0] / 255).sum() / pxs

            if ratio <= 0.7:
                continue

            img_a = equ_a.GetPerspective(fov, horizon, azimuth, height, width)
            img_s = equ_s.GetPerspective(fov, horizon, azimuth, height, width)
            img_o = equ_o.GetPerspective(fov, horizon, azimuth, height, width)

            pil_e = Image.fromarray(img_e)
            pil_a = Image.fromarray(img_a)
            pil_s = Image.fromarray(img_s)
            pil_o = Image.fromarray(img_o)

            pil_e.save(os.path.splitext(os.path.join(fov_dir, mode, 'environment', environment_list[i]))[0]+'_'+str(count)+'.png', 'png')
            pil_a.save(os.path.splitext(os.path.join(fov_dir, mode, 'albedo', albedo_list[i]))[0]+'_'+str(count)+'.png', 'png')
            pil_s.save(os.path.splitext(os.path.join(fov_dir, mode, 'shading', shading_list[i]))[0]+'_'+str(count)+'.png', 'png')
            pil_o.save(os.path.splitext(os.path.join(fov_dir, mode, 'original', original_list[i]))[0]+'_'+str(count)+'.png', 'png')
            
            count += 1

        