import bpy
from math import radians
import random
import numpy as np
import os
import glob
import mathutils
import uuid

PIECE_BASE_NAME = 'Piece'


def rotateObject(obj, axis, degrees):
    mat = mathutils.Matrix.Rotation(radians(degrees), 4, axis)
    T = obj.matrix_world.copy()
    NEWT = T @ mat
    obj.matrix_world = NEWT

def translateObject(obj, vec):
    mat = mathutils.Matrix.Translation(vec)
    T = obj.matrix_world.copy()
    NEWT = T @ mat
    obj.matrix_world = NEWT    


def test():
    bpy.data.images.load("/Users/daniele/Desktop/TM_test/abstract-3557682_960_720.jpg").name = "screen_1"

    obj_material = bpy.data.materials['Texture1']


    obj_material.node_tree.nodes['Image Texture'].image = bpy.data.images['screen_1']

    print("OK")
    
def browseFiles(folderpath, extensions):
    if not isinstance(extensions,list):
        extensions = [extensions]
    files = []
    for ext in extensions:
        files += glob.glob(os.path.join(folderpath,"*.{}".format(ext)))
    return files
    
    
def loadPly(filepath, rescale_factor = 0.005, collection='Items'):
    bpy.ops.object.select_all(action='DESELECT')
    model = bpy.ops.import_mesh.ply(filepath=filepath)
    
    name = None
    if len(bpy.context.selected_objects)>0:
        name = bpy.context.selected_objects[0].name
        
    bpy.ops.object.select_all(action='DESELECT')
        
    print("LOADING",name)
    if name is not None:
        obj = bpy.data.objects[name]
        obj.name = PIECE_BASE_NAME
        obj.active_material = bpy.data.materials['SeedMaterial'].copy()
        obj.dimensions = obj.dimensions * rescale_factor
        try:
            bpy.data.collections[collection].objects.link(obj)
        except:
            pass
        return obj
    return None
   
def assignRandomTextureToObject(obj, images_list):
    obj_material = obj.active_material
    image = random.choice(images_list)
    obj_material.node_tree.nodes['Image Texture'].image = image


def deletePieces():
    bpy.ops.object.select_all(action='DESELECT')
    
    for obj in bpy.data.objects:
        if PIECE_BASE_NAME in obj.name:
            obj.select_set(True)

    bpy.ops.object.delete()
    

def randomizeCameraPosition():
    camera = bpy.data.objects['Camera']
    camera.location = (
        0.0,
        0.0,
        random.uniform(0.2,0.7)
    )

def randomizeLight():
    light = bpy.data.objects['Point']
    bounds = np.array([
        [-0.15,0.15],
        [-0.15,0.15],
        [0.3,0.6]
    ])
    
    light.location = (
        random.uniform(bounds[0][0],bounds[0][1]),
        random.uniform(bounds[1][0],bounds[1][1]),
        random.uniform(bounds[2][0],bounds[2][1])
    )
    l = bpy.data.lights['Point']
    l.energy = random.uniform(1,15)


def randomizeObjectPose(obj):

    bounds = np.array([
        [-0.2,0.2],
        [-0.15,0.15],
        [0.02,0.15]
    ])
    
    r0 = random.uniform(-90,90)
    r1 = random.uniform(-90,90)
    r2 = random.uniform(-90,90)
    obj.rotation_euler = (r0,r1,r2) 
    obj.location = (
        random.uniform(bounds[0][0],bounds[0][1]),
        random.uniform(bounds[1][0],bounds[1][1]),
        random.uniform(bounds[2][0],bounds[2][1])
    )
    print("BEFORE", obj.scale)
    obj.scale = obj.scale * random.uniform(0.3,1.05)
    print("AFTER", obj.scale)
    
    
def randomizeTable():
    table = bpy.data.objects['Table']
    assignRandomTextureToObject(table,table_textures)
    r2 = random.uniform(-90,90)
    table.rotation_euler = (0,0,r2) 
    
    
def clearUnusedImages():
    """ CLEARS UNUSED IMAGES IN THE REPOSITORY """
    for image in bpy.data.images:
        if image.users:
            continue
        bpy.data.images.remove(image)  


# bpy.context.scene.cycles.device = 'GPU'
bpy.data.scenes["Scene"].render.resolution_x = 640
bpy.data.scenes["Scene"].render.resolution_y = 480
bpy.data.scenes["Scene"].cycles.samples = 200
bpy.data.scenes["Scene"].cycles.max_bounces= 1
bpy.data.scenes["Scene"].cycles.min_bounces = 1
bpy.data.scenes["Scene"].render.tile_x = 64
bpy.data.scenes["Scene"].render.tile_y = 64
bpy.data.scenes["Scene"].cycles.device = 'GPU'

clearUnusedImages()

# LOAD OBJECT TEXTURES
objects_texture_path = 'media/ObjectsTexture'
objects_textures_files = browseFiles(objects_texture_path, ['jpg','png'])
objects_textures = []

for index, texture_file in enumerate(objects_textures_files):
    txt = bpy.data.images.load(texture_file)
    txt.name = "object_texture_{}".format(index)
    objects_textures.append(txt)


# LOAD TABLE TEXTURES
table_texture_path = 'media/TableTextures'
table_textures_files = browseFiles(table_texture_path, ['jpg','png'])
table_textures = []

for index, texture_file in enumerate(table_textures_files):
    txt = bpy.data.images.load(texture_file)
    txt.name = "table_texture_{}".format(index)
    table_textures.append(txt)



# LOAD MODELS
models_extension = 'ply'
models_path = "media/CadModels/"
models_files = glob.glob(os.path.join(models_path,"*.{}".format(models_extension)))



def createOutputNode(output_path='/tmp/gino', name='OutFile', format='OPEN_EXR'):
    f = nodes.new('CompositorNodeOutputFile')
    f.label = name
    f.name = name
    f.format.file_format = format
    f.base_path = output_path

    #f.file_slots.new("Depth")
    return f


output_path = '/tmp/vc_dataset_train/frame_{}/'
for j in range(500):
    
    # RANDOMIZE TABLE
    randomizeTable()
    #table = bpy.data.objects['Table']
    #assignRandomTextureToObject(table,table_textures)
        

    # RANDOMIZE OBJECTS
    deletePieces()
    for i in range(20):
        model = loadPly(
            random.choice(models_files),
            rescale_factor = 0.0007
        )
        randomizeObjectPose(model)
        assignRandomTextureToObject(model,objects_textures)
        
    
    # RANDOMIZE CAMERA AND LIGHTS
    randomizeCameraPosition()
    randomizeLight()


    # RENDER OUTPUT
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    nodes = tree.nodes
    links = tree.links
    render = nodes["Render Layers"]
    out_folder = output_path.format(uuid.uuid4().hex)

    depth_node = createOutputNode(output_path=out_folder,format='OPEN_EXR')
    rgb_node = createOutputNode(output_path=out_folder,format='JPEG')

    links.new(
        render.outputs['Image'],
        rgb_node.inputs['Image']
    )
    links.new(
        render.outputs['Depth'],
        depth_node.inputs['Image']
    )


    camera = bpy.data.objects["Camera"]
    for i in range(11):
        print("#"*20)


        camera.location.x += 0.002
        #rotateObject(camera, 'Y', -0.25)
            
        num = str(i).zfill(5)
        
        depth_node.file_slots[0].path = 'depth_{}_'.format(num)
        rgb_node.file_slots[0].path = 'rgb_{}_'.format(num)
        bpy.ops.render.render(write_still=True)



    nodes.remove(depth_node)
    nodes.remove(rgb_node)
