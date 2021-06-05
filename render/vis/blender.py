import bpy
import mathutils
import numpy as np


DRONE_FBX = ''
TRAJ_PATH = ''

boxes = []
pillars = []
drones = []

def look_at(camera, focus_point):
    looking_direction = camera.location - mathutils.Vector(focus_point.tolist())
    rot_quat = looking_direction.to_track_quat('Z', 'Y')

    camera.rotation_euler = rot_quat.to_euler()
    # camera.location = rot_quat * mathutils.Vector((0.0, 0.0, distance))

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

class Drone(object):
    """docstring for Drone"""
    def __init__(self, scale=1., name=None):
        super(Drone, self).__init__()
        # gen a drone with initial pose
        file_loc = DRONE_FBX
        bpy.ops.import_scene.fbx(filepath=file_loc)
        obs = bpy.context.selected_objects
        assert len(obs) == 5
        for ob in obs:
            if 'drone' in ob.name:
                self.drone = ob
        self.angle = 0.
        if name is not None:
            self.drone.name = name

    def keyframe_insert(self, pose, frameid):
        self.drone.location = pose[:3].tolist()
        self.drone.keyframe_insert(data_path="location", frame=frameid)
        self.drone.rotation_mode = 'ZXY'
        self.drone.rotation_euler = (pose[-2], pose[-1], 0)
        self.drone.keyframe_insert(data_path="rotation_euler", frame=frameid)
        self.angle = frameid * 2 * np.pi / 10.
        for rotor in self.drone.children:
            rotor.rotation_mode = 'XYZ'
            rotor.rotation_euler.y = self.angle
            rotor.keyframe_insert(data_path="rotation_euler", frame=frameid)

import pickle
d = pickle.load(open(TRAJ_PATH, 'rb'))

traj = d['traj'] # T x N x 8
traj = traj[:1000,:,:]
tmp = traj[:,:,-1].copy()
traj[:,:,-1] = traj[:,:,-2] + np.pi
traj[:,:,-2] = tmp - np.pi/2
T = traj.shape[0]
num_agents = traj.shape[1]
obstacles = d['obs']
frame_rate_mult = 1

scene = bpy.data.scenes["Scene"]
# Set render resolution
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080

cam = scene.camera
# Set camera fov in degrees
cam.data.angle = 120*(np.pi/180.0)

cam.rotation_mode = 'XYZ'
cam.rotation_euler = (np.pi/2, 0, 0)
cam.location.x = 15
cam.location.y = -10
cam.location.z = 10

# for idx_a in range(num_agents):
#     add_a_curve(traj[:,idx_a,:3], np.random.rand(3).tolist()+[1.,], 'traj_%d'%idx_a)

for idx_a in range(num_agents):
    drones.append(Drone('drone_%d'%idx_a))

for idx_t in range(T):
    if idx_t%10 > 0:
        continue
    # cam.location.y = traj[idx_t,:,1].min() - 10
    # cam.keyframe_insert(data_path="location", frame=idx_t*frame_rate_mult+1)
    camloc = idx_t-300 if idx_t >= 300 else 0
    cam.location = traj[camloc,:,:3].mean(axis=0).tolist()
    point_to_look_at = traj[idx_t,:,:3].mean(axis=0)
    look_at(cam, point_to_look_at)
    cam.keyframe_insert(data_path="location", frame=idx_t*frame_rate_mult+1)
    cam.keyframe_insert(data_path='rotation_euler', frame=idx_t*frame_rate_mult+1)

    for idx_a in range(num_agents):
        pose = traj[idx_t,idx_a,[0,1,2,-2,-1]]
        drones[idx_a].keyframe_insert(pose, idx_t*frame_rate_mult+1)

# for exporting all boxes
#import bpy

#scene = bpy.context.scene
#boxes = [ob2box(obj) for obj in scene.objects if 'box' in obj.name]

#np.savez('obs.npz', boxes=boxes)
