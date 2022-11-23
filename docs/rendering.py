import bpy
import numpy as np
import math
import pandas as pd
import os
# add on
bl_info = {   
    "name": "Rendering",
    "author": "Yonghee Oh",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Add > Mesh > New Object",
    "description": "Adds a new Mesh Object",
    "warning": "",
    "doc_url": "",
    "category": "Add Mesh",
}

# global variables
pwd = '/home/cvar/yonghee/oie'
model_name = "grid1"
animation_end = 1


os.chdir("/home/cvar/yonghee/oie")
os.chdir("./exr")
days = os.listdir(os.path.join(os.getcwd() , "LavalSkyDBenvmap"))
times = list(range(0,len(days), 1))
for day in range(len(days)) :
    times[day] = os.listdir(os.path.join(os.path.join(os.getcwd() , "LavalSkyDBenvmap") , days[day]))

exr_path_memory= []
label_path_memory = []
image_path_memory1 = [] 
image_path_memory2 = []

for i in range(len(days)):
    for time in times[i]:
        exr_path_memory.append( pwd+ '/exr/LavalSkyDBenvmap/'+days[i]+'/'+time+'/envmap.exr')
        label_path_memory.append(  pwd+ "/exr/LavalSkyDBcsvday/" + days[i] )
        image_path_memory1.append(pwd+'/model/'+model_name+'/'+model_name+'/'+days[i])
        image_path_memory2.append(pwd+'/model/'+model_name+'/'+model_name+'/'+days[i]+'/'+time)
        
os.chdir("../")


# make scene
def modeling():
    pass 
def load_data(input_path):
    pass

def animation(input_data):
    pass


def environment_map(exr_path_memory):

    image = bpy.data.images.load(filepath = exr_path_memory)
    bpy.data.worlds["World"].node_tree.nodes["Environment Texture"].image = image






def save_data(params ,image_path_memory1,image_path_memory2, exr_path_memory ):
    
    params.insert(0,'exr_path_memory',exr_path_memory) 
    params.insert(0,'path_environment',list('../'+model_name+'/environment/environment'+str(i).zfill(4)+ '.png' for i in range(bpy.context.scene.frame_end+1))) 
    params.insert(0,'path_shading',list('../'+model_name+'/shading/shading'+str(i).zfill(4)+ '.png' for i in range(bpy.context.scene.frame_end+1))) 
    params.insert(0,'path_albedo',list('../'+model_name+'/albedo/albedo'+str(i).zfill(4)+ '.png' for i in range(bpy.context.scene.frame_end+1))) 
    params.insert(0,'path_original',list('../'+model_name+'/original/original'+str(i).zfill(4)+ '.png' for i in range(bpy.context.scene.frame_end+1))) 
    
    os.makedirs(image_path_memory1, exist_ok =True)
    os.mkdir(image_path_memory2)
    #os.mkdir("./model/"+model_name+'/'+model_name)
    #os.mkdir("./model/"+model_name+'/'+model_name+'/'+days[ch_env])
    
    
    
    params.to_csv(image_path_memory2+'/output.csv' , index = False)

def settings():
    
    # render properties
 
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.render.fps = 1
    bpy.context.scene.render.resolution_x = 192  # will be changed
    bpy.context.scene.render.resolution_y = 108  # will be changed
    
    
    bpy.context.scene.cycles.max_bounces = 8
    bpy.context.scene.cycles.diffuse_bounces = 4
    bpy.context.scene.cycles.glossy_bounces = 6
    bpy.context.scene.cycles.transmission_bounces = 8
    bpy.context.scene.cycles.volume_bounces = 2
    bpy.context.scene.cycles.transparent_max_bounces = 8
    
    bpy.context.scene.cycles.film_exposure = 1
    
    
    
    
    # output properties
    
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    bpy.context.scene.render.image_settings.color_depth = '8'
    bpy.context.scene.render.image_settings.compression = 15

    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.render.use_sequencer = True
    
    bpy.context.scene.render.use_overwrite = True
    
    
    
    # view layer properties 
    bpy.context.scene.view_layers["ViewLayer"].use_pass_diffuse_direct = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_diffuse_indirect = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_diffuse_color = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_environment = True

    bpy.context.scene.render.use_single_layer = True
    
    # world properties
    
    bpy.context.scene.world.cycles_visibility.camera = True
    bpy.context.scene.world.cycles_visibility.diffuse = True
    bpy.context.scene.world.cycles_visibility.glossy = True
    bpy.context.scene.world.cycles_visibility.transmission = True
    bpy.context.scene.world.cycles_visibility.scatter = True
    
    
    
    # camera properties
    
    bpy.context.object.data.type = 'PANO'
    bpy.context.object.data.cycles.panorama_type = 'EQUIRECTANGULAR'
    
    bpy.context.object.data.cycles.latitude_min = -1.5708
    bpy.context.object.data.cycles.latitude_max = 1.5708
    bpy.context.object.data.cycles.longitude_min = -3.14159
    bpy.context.object.data.cycles.longitude_max = 3.14159
    bpy.context.object.data.shift_x = 0
    bpy.context.object.data.shift_y = 0
    bpy.context.object.data.clip_start = 0.01
    bpy.context.object.data.clip_end = 10000
    
    
    
    # scene properties
    bpy.context.scene.frame_current = 0
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = animation_end






    
def compositing(image_path_memory2):
    

        nodesField = bpy.context.scene.node_tree
        for currentNode in nodesField.nodes:
            nodesField.nodes.remove(currentNode)

        
        scene = bpy.context.scene
        nodes = scene.node_tree.nodes

        
        render_layers =  nodes.new('CompositorNodeRLayers')
        mix_add = nodes.new('CompositorNodeMixRGB')
        mix_add.blend_type = 'ADD'
        
        math_grater = nodes.new('CompositorNodeMath')
        math_grater.operation = 'GREATER_THAN'
        math_grater.inputs[1].default_value = 0
        
        math_less = nodes.new('CompositorNodeMath')
        math_less.operation = 'LESS_THAN'
        math_less.inputs[1].default_value = 0.5
 

        
        output_file = nodes.new("CompositorNodeOutputFile")

        output_file.base_path = image_path_memory2 + '/'
        

        scene.node_tree.links.new(
            render_layers.outputs['DiffDir'],
            mix_add.inputs[1]
        )
        scene.node_tree.links.new(
            render_layers.outputs['DiffInd'],
            mix_add.inputs[2]
        )
        scene.node_tree.links.new(
            render_layers.outputs['Env'],
            math_grater.inputs[0]
        )
        scene.node_tree.links.new(
            math_grater.outputs['Value'],
            math_less.inputs[0]
        )
        
        

        
        output_file.file_slots.remove(output_file.inputs[0])
        output_file.file_slots.new("original")
        output_file.file_slots.new("shading")
        output_file.file_slots.new("albedo")
        output_file.file_slots.new("environment")
        
        output_file.file_slots[0].path = "original/original"
        output_file.file_slots[1].path = "shading/shading"
        output_file.file_slots[2].path = "albedo/albedo"
        output_file.file_slots[3].path = "environment/environment"
        
        scene.node_tree.links.new(
            render_layers.outputs['Image'],
            output_file.inputs['original']
        )
        scene.node_tree.links.new(
            mix_add.outputs['Image'],
            output_file.inputs['shading']
        )
        scene.node_tree.links.new(
            render_layers.outputs['DiffCol'],
            output_file.inputs['albedo']
        )
        scene.node_tree.links.new(
            math_less.outputs['Value'],
            output_file.inputs['environment']
        )
        
        
        
        
def render_and_save(animation = False ):
    
    if(animation==False):
        bpy.context.scene.render.image_settings.file_format = 'PNG'
   
    else :
        bpy.context.scene.render.image_settings.file_format = 'AVI_JPEG'




    bpy.ops.render.render(use_viewport = True,write_still = True, animation = True)     
    

def extract_camera_params(frames,label_path_memory):
    camera_position = np.zeros((frames.shape[0],15))
    row_name = frames
    col_name = ['loc_x','loc_y','loc_z','rot_x','rot_y','rot_z',  'elevation','azimuth' ,  'camera_latitude_min' ,'camera_latitude_max','camera_longitude_min','camera_longitude_max',  'px','py','exposure']
    
    df = pd.read_csv(label_path_memory)
    df = df[["Sun elevation" , "Sun azimuth"]]
    
    
    for i in range(frames.shape[0]):
        
        bpy.context.scene.frame_set(frames[i])
        
        # camera extrinsic params
        camera_position[i,0] = bpy.context.object.location[0]
        camera_position[i,1] = bpy.context.object.location[1]
        camera_position[i,2] = bpy.context.object.location[2]
        camera_position[i,3] =  math.degrees(bpy.context.object.rotation_euler[0])
        camera_position[i,4] =  math.degrees(bpy.context.object.rotation_euler[1])
        camera_position[i,5] =  math.degrees(bpy.context.object.rotation_euler[2])
        
        # sun position
        camera_position[i,6] = df.iloc[0,0]
        camera_position[i,7] = df.iloc[0,1]
        
        # camera intrinsic params
        camera_position[i,8] = bpy.context.object.data.cycles.latitude_min
        camera_position[i,9] = bpy.context.object.data.cycles.latitude_max
        camera_position[i,10] = bpy.context.object.data.cycles.longitude_min
        camera_position[i,11] = bpy.context.object.data.cycles.longitude_max
        camera_position[i,12] = bpy.context.object.data.shift_x
        camera_position[i,13] = bpy.context.object.data.shift_y
        
        
        # rendering setting
        camera_position[i,14] = bpy.context.scene.view_settings.exposure

        
        
        
        

    df = pd.DataFrame(camera_position, index = row_name , columns = col_name)    
    
    return df 
    

def main(context):   # main 
    
    
    
    """
    camera location and animation should be set by user.
    """
    #input_data = load_data("D:\\blend\\city1\\uploads_files_702403_BLEND\\BLEND\\data\\camera3_input.csv")
    #animation(input_data)  scale different , so  frames different

    
    # initial setting
    settings()
    
    
    for ch_env in range(len(exr_path_memory)):
        #environmentamp
        environment_map(exr_path_memory[ch_env])

    
         #in camera animation , extract camera parameter and save it.
        params = extract_camera_params(np.array(list(i for i in range(bpy.context.scene.frame_end+1))),label_path_memory[ch_env])
        save_data(params, image_path_memory1[ch_env] ,image_path_memory2[ch_env] , exr_path_memory[ch_env])
    
        # composite intrinsic image and rendering and then save it.
        compositing(image_path_memory2[ch_env])
        render_and_save( animation =False)
    

    
    
    
    
    
    
    


class RenderingOperator(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "object.rendering_operator"   # class number  like  person id 
    bl_label = "Rendering Object Operator"  # search operator name    F3 > operator 
 
    @classmethod
    def poll(cls, context):    # is this object ? 
        return context.active_object is not None

    def execute(self, context):    # execute 
        main(context)
        return {'FINISHED'}





def menu_func(self, context):   # add menu 
    self.layout.operator(RenderingOperator.bl_idname, text=RenderingOperator.bl_label)


# Register and add to the "object" menu (required to also use F3 search "Simple Object Operator" for quick access).
def register():
    bpy.utils.register_class(RenderingOperator)  # register class / function in blender 
    bpy.types.TOPBAR_MT_render.append(menu_func)

def unregister():
    #bpy.utils.unregister_class(RenderingOperator)     # unregister class /funtion in blender 
    bpy.types.TOPBAR_MT_render.remove(menu_func)


if __name__ == "__main__":
    register()
    
    # test call
    #bpy.ops.object.rendering_operator()
 
