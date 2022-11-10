# Dataset Creation

## Install Blender
You can install blender here : 
https://www.blender.org/
We use Blender 3.2ver.


## Add-ons
### Sun-aligner
To determine the position of the sun in the hdri, we use Sun-aligner. 
https://github.com/akej74/hdri-sun-aligner 

The position of the sun is determined based on the image pixel value.

Edit - Preference - Add ons - open zip file

### Our python script
Download python script. [here] (https://github.com/NahyukLEE/OIEID/blob/main/docs/rendering.py)



## Install Package
Install pandas in '~/BlenderFoundation/Blender/python/lib/site-packages'
To use pip installation , you can try
```
import subprocess
import sys
import os

python_exe = os.path.join(sys.prefix, 'bin', 'python.exe')
subprocess.call([python_exe, "-m", "ensurepip"])
subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
subprocess.call([python_exe, "-m", "pip", "install", "pandas"])
```
or
https://www.youtube.com/watch?v=gyRoY9QUNg0&t=171s


## Model setting
### Get Assets
1. 3dsky
2. CGtrader https://www.cgtrader.com/
3. Turbosquid
4. Blendswap
5. Chocofur
6. Free3d
7. Sketchfab

You can download free assets including 'asset.blend' and 'textrue'.

If you can't see 'textures' , then use 'find missing files'. 


---
Actually, we use reconstructed object from https://vcc.tech/UrbanScene3D.


### Get HDRI
poly haven https://polyhaven.com/

You can download free hdris.

In Blender World Properties , select environment texture and upload HDRI.

--- 
Actually , we use LAVAL outdoor hdr data.


## Camera setting
Add camera ( shift + A ) and transformate using Grap, Rotation, Scaling and go to Animation Mode.

Register the keyframe(I) by moving the camera along the path you want.

Consider the physical elements and fps while making the animation.
- distance(m) = speed(m/s) * time(s)
- fps = frame(f) / time(s)

--- 
Actually, we use camera position get from https://vcc.tech/UrbanScene3D.

## Parameters

You can set many parameters in Blender-properties or our python script.

In current our experiment , Mandatory parameters are

- engine = 'CYCLES'
- device = 'GPU'
- type = 'PANO'
- panorama_type = 'EQUIRECTANGULAR'

In current our experiment , changeable parameters are

- resolution_x = 1280
- resolution_y = 720 
- cycles.samples = 256
- render.fps = 1
- frame_start = 0
- frame_end = 200
- file_format = 'PNG'

Every parameter is set by python method.



## Run

You should change 'model_name' and select camera which is used.

Run the script , and go to Render Mode > rendering




TBU : input camera parameter , laval hdr dataset  




