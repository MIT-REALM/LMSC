import bpy
import mathutils
import numpy as np

boxes = []
pillars = []
drones = []

def look_at(camera, focus_point):
    looking_direction = camera.location - mathutils.Vector(focus_point.tolist())
    rot_quat = looking_direction.to_track_quat('Z', 'Y')

    camera.rotation_euler = rot_quat.to_euler()
    # camera.location = rot_quat * mathutils.Vector((0.0, 0.0, distance))

def add_a_box(box):
    box = np.array(box)
    sx, sy, sz = ((box[3:] - box[:3])/2).tolist()
    x, y, z = ((box[3:] + box[:3])/2).tolist()
    # gen a box with pos and scale
    bpy.ops.mesh.primitive_cube_add(location=(x,y,z))
    ob = bpy.context.selected_objects[0]
    ob.scale = (sx, sy, sz)
    return ob

def ob2box(ob):
    scale = np.array(ob.scale)
    location = np.array(ob.location)
    box = np.concatenate([location - scale, location + scale])
    return box

def add_a_curve(coords, color, name):
    mat = bpy.data.materials.new(name='Materials_'+name)
    # mat.use_nodes = False
    # mat.inputs['Base Color'].default_value = color
    mat.diffuse_color = color

    # create the Curve Datablock
    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = 2

    # map coords to spline
    polyline = curveData.splines.new('POLY')
    polyline.points.add(coords.shape[0]-1)
    for i in range(coords.shape[0]):
        x, y, z = coords[i][0], coords[i][1], coords[i][2]
        polyline.points[i].co = (x, y, z, 1)

    # create Object
    curveOB = bpy.data.objects.new(name, curveData)
    curveData.bevel_depth = 0.1

    # attach to scene and validate context
    bpy.context.collection.objects.link(curveOB)
    if curveOB.data.materials:
        curveOB.data.materials[0] = mat
    else:
        curveOB.data.materials.append(mat)
    return curveOB

def add_a_material(texture, name):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    links = mat.node_tree.links
    nodes = mat.node_tree.nodes
    # create nodes
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    tex_image = nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(texture)
    tex_coordinate = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs['Scale'].default_value = (10,10,10)
    links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
    links.new(tex_image.inputs["Vector"], mapping.outputs["Vector"])
    links.new(mapping.inputs["Vector"], tex_coordinate.outputs["UV"])
    return mat

# def add_a_pillar(pos):
#     file_loc = ''
#     imported_object = bpy.ops.import_scene.obj(filepath=file_loc)
#     obj_object = bpy.context.selected_objects[0]
#     print('Imported name: ', obj_object.name)
#     # set pos and scale
#     return obj_object

def add_a_drone(pose):
    # gen a drone with initial pose
    file_loc = '/home/daweis2/Downloads/Drone_Costum/Material/drone_costum.obj'
    bpy.ops.import_scene.obj(filepath=file_loc)
    ob = bpy.context.selected_objects[0]
    ob.rotation_mode = 'ZXY'
    ob.scale = (0.1, 0.1, 0.1)
    # ob.scale = (0.05, 0.05, 0.05)
    ob.location = pose[:3].tolist()
    ob.rotation_euler = (pose[-2], pose[-1], 0)
    return ob

import pickle
d = pickle.load(open('/home/daweis2/ccmcbf/data/data_16agents_noise_1_sepInit.pkl', 'rb'))

traj = d['traj'] # T x N x 8
# traj = traj[:20000,:,:]
tmp = traj[:,:,-1].copy()
traj[:,:,-1] = traj[:,:,-2]
traj[:,:,-2] = tmp+np.pi/2
T = traj.shape[0]
num_agents = traj.shape[1]
obstacles = d['obs']
frame_rate_mult = 1

mat_bricks = add_a_material('/home/daweis2/ccmcbf/data/texture/bricks.jpg', 'Materials_bricks')
mat_rock = add_a_material('/home/daweis2/ccmcbf/data/texture/rock.jpg', 'Materials_rock')
mat_cement = add_a_material('/home/daweis2/ccmcbf/data/texture/cement.png', 'Materials_cement')

# create light datablock, set attributes
light_data = bpy.data.lights.new(name="light_for_camera", type='POINT')
light_data.energy = 3e6
light_data.specular_factor = 0.
# create new object with our light datablock
light_object = bpy.data.objects.new(name="light_for_camera", object_data=light_data)
# link light object
bpy.context.collection.objects.link(light_object)
# make it active 
bpy.context.view_layer.objects.active = light_object

rocks_idx = list(range(0, 5)) + list(range(17, 22)) + list(range(27, 32))
cement_idx = list(range(5, 10)) + list(range(12, 17)) + list(range(22, 27))

for i, obs in enumerate(obstacles):
    boxes.append(add_a_box(obs))
    boxes[-1].name = 'box_%d'%i
    ob = boxes[-1]

    mat = mat_bricks
    if i in rocks_idx:
        mat = mat_rock
    if i in cement_idx:
        mat = mat_cement
    # Assign it to object
    if ob.data.materials:
        ob.data.materials[0] = mat
    else:
        ob.data.materials.append(mat)

scene = bpy.data.scenes["Scene"]
# Set render resolution
scene.render.resolution_x = 1920*4
scene.render.resolution_y = 1080*4

cam = scene.camera
# Set camera fov in degrees
# cam.data.angle = 120*(np.pi/180.0)

cam.rotation_mode = 'XYZ'
cam.rotation_euler = (np.array([68.2, -1.88, 112])/180*np.pi).tolist()
cam.location.x = 263.28
cam.location.y = 41.393
cam.location.z = 75.764
cam.data.lens = 25
cam.data.clip_end = 1000
light_object.location.x = 263.28
light_object.location.y = 41.393
light_object.location.z = 75.764

for idx_a in range(num_agents):
    add_a_curve(traj[:,idx_a,:3], np.random.rand(3).tolist()+[1.,], 'traj_%d'%idx_a)

