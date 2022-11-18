import bpy
import numpy as np
import math
import pandas as pd
bl_info = {   # add on 
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





model_name = "purchase"



path_name = './yonghee/oie/blender/model/'+model_name+'/data/'




def modeling():
    pass 

def environment_map():
    pass




    
def load_data(input_path):
    return pd.read_csv(input_path)


def save_data(params , output_path):
    
    params.insert(0,'path_environment',list(output_path+'environment/environment'+str(i).zfill(4)+ '.png' for i in range(bpy.context.scene.frame_end+1))) 
    params.insert(0,'path_shading',list(output_path+'shading/shading'+str(i).zfill(4)+ '.png' for i in range(bpy.context.scene.frame_end+1))) 
    params.insert(0,'path_albedo',list(output_path+'albedo/albedo'+str(i).zfill(4)+ '.png' for i in range(bpy.context.scene.frame_end+1))) 
    params.insert(0,'path_original',list(output_path+'original/original'+str(i).zfill(4)+ '.png' for i in range(bpy.context.scene.frame_end+1))) 
    
    
    
    params.to_csv(output_path+'output.csv' , index = False)


def settings():
    
    # render properties
 
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.render.fps = 1
    bpy.context.scene.render.resolution_x = 4096
    bpy.context.scene.render.resolution_y = 2160
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'


    bpy.context.scene.use_nodes = True

    
    
    # camera properties
    
    bpy.context.object.data.type = 'PANO'
    bpy.context.object.data.cycles.panorama_type = 'EQUIRECTANGULAR'
    

    

    bpy.context.scene.render.use_single_layer = True
    bpy.context.scene.render.use_overwrite = True
    bpy.context.scene.render.use_compositing = True
    
    #bpy.context.scene.view_layers["View Layer"].use_pass_diffuse_direct = True
    #bpy.context.scene.view_layers["View Layer"].use_pass_diffuse_indirect = True
    #bpy.context.scene.view_layers["View Layer"].use_pass_diffuse_color = True
    #bpy.context.scene.view_layers["View Layer"].use_pass_environment = True


    bpy.context.scene.world.cycles_visibility.camera = True
    bpy.context.scene.world.cycles_visibility.diffuse = True
    bpy.context.scene.world.cycles_visibility.glossy = True
    bpy.context.scene.world.cycles_visibility.transmission = True
    bpy.context.scene.world.cycles_visibility.scatter = True

    bpy.context.scene.use_nodes = True

#def animation(input_data):
#    LRS = input_data
#    for i  in range(LRS.shape[0]):
#        #bpy.context.scene.frame_current = i*20
#        bpy.context.object.location[0] = LRS.iloc[i,0]
#        bpy.context.object.location[1] = LRS.iloc[i,1]
#        bpy.context.object.location[2] = LRS.iloc[i,2]
#        bpy.context.object.rotation_euler[0] = math.radians(int(LRS.iloc[i,3]))
#        bpy.context.object.rotation_euler[1] = math.radians(int(LRS.iloc[i,4]))
#        bpy.context.object.rotation_euler[2] = math.radians(int(LRS.iloc[i,5]))
#        #bpy.ops.anim.keyframe_insert_by_name(type="LocRotScale")
#        bpy.context.object.keyframe_insert(data_path='location', frame = i*20)
#        bpy.context.object.keyframe_insert(data_path='rotation_euler', frame = i*20)
        
    
def compositing(path):
        scene = bpy.context.scene
        nodes = scene.node_tree.nodes

        render_layers = nodes['Render Layers']
        mix_add = nodes.new('CompositorNodeMixRGB')
        mix_add.blend_type = 'ADD'
        
        math_grater = nodes.new('CompositorNodeMath')
        math_grater.operation = 'GREATER_THAN'
        math_grater.inputs[1].default_value = 0
        
        math_less = nodes.new('CompositorNodeMath')
        math_less.operation = 'LESS_THAN'
        math_less.inputs[1].default_value = 0.5
 

        output_file = nodes.new("CompositorNodeOutputFile")

        output_file.base_path = path


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
        
def render_and_save(path, animation = False):

    
    if(animation==False):
        bpy.context.scene.render.image_settings.file_format = 'PNG'
   
    else :
        bpy.context.scene.render.image_settings.file_format = 'AVI_JPEG'
        
        
        
    

    #bpy.ops.render.render(use_viewport = True,write_still = True, animation = True)  
    bpy.ops.render.render(use_viewport = True,animation = True)   
    
    # original 
    #bpy.context.scene.render.filepath = path+'original\\'
    #bpy.ops.render.render(use_viewport = True,write_still = True, animation = True) 
    # albedo
    #bpy.context.scene.render.filepath = path+'albedo\\'
    #bpy.ops.render.render(use_viewport = True,write_still = True, animation = True) 
    #shading
    #bpy.context.scene.render.filepath = path+'shading\\'
    #bpy.ops.render.render(use_viewport = True,write_still = True, animation = True) 

    



def extract_camera_params(frames):
    camera_position = np.zeros((frames.shape[0],16))
    row_name = frames
    col_name = ['loc_x','loc_y','loc_z','rot_x','rot_y','rot_z',   'camera_latitude_min' ,'camera_latitude_max','camera_longitude_min','camera_longitude_max',  'px','py','exposure','gamma','longitude','latitude']
    
    for i in range(frames.shape[0]):
        
        bpy.context.scene.frame_set(frames[i])
        
        # camera extrinsic params
        camera_position[i,0] = bpy.context.object.location[0]
        camera_position[i,1] = bpy.context.object.location[1]
        camera_position[i,2] = bpy.context.object.location[2]
        camera_position[i,3] =  math.degrees(bpy.context.object.rotation_euler[0])
        camera_position[i,4] =  math.degrees(bpy.context.object.rotation_euler[1])
        camera_position[i,5] =  math.degrees(bpy.context.object.rotation_euler[2])
        
        # camera intrinsic params
        camera_position[i,6] = bpy.context.object.data.cycles.latitude_min
        camera_position[i,7] = bpy.context.object.data.cycles.latitude_max
        camera_position[i,8] = bpy.context.object.data.cycles.longitude_min
        camera_position[i,9] = bpy.context.object.data.cycles.longitude_max
        camera_position[i,10] = bpy.context.object.data.shift_x
        camera_position[i,11] = bpy.context.object.data.shift_y
        
        
        # rendering setting
        #camera_position[i,6] = math.degrees(bpy.context.object.data.cycles.fisheye_fov)
        camera_position[i,12] = bpy.context.scene.view_settings.exposure
        camera_position[i,13] = bpy.context.scene.view_settings.gamma
        
        
        
        # sun position
        camera_position[i,14] = bpy.context.scene.hdri_sa_props.long_deg
        camera_position[i,15] = bpy.context.scene.hdri_sa_props.lat_deg

    df = pd.DataFrame(camera_position, index = row_name , columns = col_name)    
    
    return df 
    

def main(context):   # main 
    
    #input_data = load_data("D:\\blend\\city1\\uploads_files_702403_BLEND\\BLEND\\data\\camera3_input.csv")
    settings()
    #animation(input_data)  scale different , so  frames different
    
    params = extract_camera_params(frames = np.array(list(i for i in range(bpy.context.scene.frame_end+1))))
    save_data(params, path_name)
    
    compositing(path = path_name)
    
    
    render_and_save(path = path_name,animation=False)
    

    
    
    
    
    
    
    


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
 
