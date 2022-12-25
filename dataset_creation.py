import os
import pandas as pd
import numpy as np
from PIL import Image
from random import uniform
from utils import *
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

modes = ['train', 'val', 'test']
split = ['albedo', 'shading', 'original', 'environment']

height, width = 240, 320
pxs = height * width

pano_dir = 'oie_data1207' # 360 panorama images will be preprocessed.
fov_dir = 'fov_dataset_new' # Output fov images will be saved in.
df = pd.read_csv(os.path.join(pano_dir,'output.csv'))

if not os.path.exists(fov_dir):
    os.makedirs(fov_dir)

for mode in modes:
    print('='*20)
    print(f'Create {mode} dataset...')
    
    fov_df = pd.DataFrame(columns=['path_original', 
                                'path_albedo',
                                 'path_shading',
                                 'path_environment',
                                 'elevation',
                                 'azimuth',
                                 'exposure'])
    
    # train/val/test split
    if mode == 'train':
        start, end = 0, int(len(df)*0.7)-1
    elif mode == 'val':
        start, end = int(len(df)*0.7), int(len(df)*0.7) + int(len(df)*0.1) -1
    elif mode == 'test':
        start, end = int(len(df)*0.7) + int(len(df)*0.1), len(df)-1

    print(start, end)
    if not os.path.exists(os.path.join(fov_dir, mode)):
        os.makedirs(os.path.join(fov_dir, mode))
        os.makedirs(os.path.join(fov_dir, mode, 'original'))
        os.makedirs(os.path.join(fov_dir, mode, 'albedo'))
        os.makedirs(os.path.join(fov_dir, mode, 'shading'))
        os.makedirs(os.path.join(fov_dir, mode, 'environment'))

    for i in tqdm(range(start, end)):
        # ground truth images for intrinsic decomposition
        path_original = os.path.join(pano_dir, df.iloc[i]['path_original'])
        path_albedo = os.path.join(pano_dir, df.iloc[i]['path_albedo'])
        path_shading = os.path.join(pano_dir, df.iloc[i]['path_shading'])
        path_environment = os.path.join(pano_dir, df.iloc[i]['path_environment'])

        # ground truth labels for outdoor illumination estimation
        elevation = df.iloc[i]['elevation']
        azimuth = df.iloc[i]['azimuth']
        
        # camera & sky parameters used in Y Hold-Geoffroy et al.(CVPR'17) 
        exposure = df.iloc[i]['exposure']
        # camera pitch(elevation)
        # fov
        # turbidity
        if not os.path.isfile(path_albedo):
            continue
        if not os.path.isfile(path_shading):
            continue
        if not os.path.isfile(path_original):
            continue
        if not os.path.isfile(path_environment):
            continue

        a = np.asarray(Image.open(path_albedo))
        s = np.asarray(Image.open(path_shading))
        o = np.asarray(Image.open(path_original))
        e = np.asarray(Image.open(path_environment))

        equ_a = Equirectangular(a)
        equ_s = Equirectangular(s)
        equ_o = Equirectangular(o)
        equ_e = Equirectangular(e)

        count = 0
        while count < 5:
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

            # save fov images
            file_name = os.path.splitext(os.path.basename(df.iloc[i]['path_environment']))[0]
            pil_e.save(os.path.join(fov_dir, mode, 'environment', file_name)+'_'+str(i)+'_'+str(count)+'.png', 'png')
            file_name = os.path.splitext(os.path.basename(df.iloc[i]['path_albedo']))[0]
            pil_a.save(os.path.join(fov_dir, mode, 'albedo', file_name)+'_'+str(i)+'_'+str(count)+'.png', 'png')
            file_name = os.path.splitext(os.path.basename(df.iloc[i]['path_shading']))[0]
            pil_s.save(os.path.join(fov_dir, mode, 'shading', file_name)+'_'+str(i)+'_'+str(count)+'.png', 'png')
            file_name = os.path.splitext(os.path.basename(df.iloc[i]['path_original']))[0]
            pil_o.save(os.path.join(fov_dir, mode, 'original', file_name)+'_'+str(i)+'_'+str(count)+'.png', 'png')
            
            fov_df = fov_df.append({'path_original':os.path.join(fov_dir, mode, 'original', os.path.splitext(os.path.basename(df.iloc[i]['path_original']))[0])+'_'+str(i)+'_'+str(count)+'.png', 
                                'path_albedo':os.path.join(fov_dir, mode, 'albedo', os.path.splitext(os.path.basename(df.iloc[i]['path_albedo']))[0])+'_'+str(i)+'_'+str(count)+'.png',
                                 'path_shading':os.path.join(fov_dir, mode, 'shading', os.path.splitext(os.path.basename(df.iloc[i]['path_shading']))[0])+'_'+str(i)+'_'+str(count)+'.png',
                                 'path_environment':os.path.join(fov_dir, mode, 'environment', os.path.splitext(os.path.basename(df.iloc[i]['path_environment']))[0])+'_'+str(i)+'_'+str(count)+'.png',
                                 # sun position
                                 'elevation':elevation,
                                 'azimuth':azimuth,
                                 # camera & sky parameters
                                 'exposure':exposure}, ignore_index=True)

            count += 1
      
        

    fov_df.to_csv(os.path.join(fov_dir, mode, 'output.csv'))