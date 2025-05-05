import os
import math
import random
import yaml
import pybullet as p
import pybullet_data
import numpy as np
import cv2
from glob import glob
from PIL import Image

class World:
    def __init__(self, soil_directory, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.soil_dir = soil_directory
        self.server_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.plane_id = p.loadURDF("plane.urdf")
        self.objects = []  # list of (body_id, class_id)

    def _load_mesh(self, mesh_path, scale=[1,1,1], mass=0.0):
        col = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_path, meshScale=scale)
        vis = p.createVisualShape(p.GEOM_MESH, fileName=mesh_path, meshScale=scale)
        return p.createMultiBody(baseMass=mass,
                                 baseCollisionShapeIndex=col,
                                 baseVisualShapeIndex=vis,
                                 basePosition=[0,0,0.05])

    def spawn_mixed_objects(self, items):
        # items: list of dicts {mesh_path, scale, class_id}
        # remove existing
        for oid, _ in self.objects:
            p.removeBody(oid)
        self.objects = []
        for itm in items:
            body_id = self._load_mesh(itm['mesh_path'], scale=[itm['scale']]*3)
            self.objects.append((body_id, itm['class_id']))

    def change_plane_texture(self):
        textures = glob(os.path.join(self.soil_dir, '*.jpg'))
        tex_id = p.loadTexture(random.choice(textures))
        p.changeVisualShape(self.plane_id, -1, textureUniqueId=tex_id)

    def randomize_objects_pose(self):
        for body_id, _ in self.objects:
            pos = [random.uniform(-0.3,0.3), random.uniform(-0.3,0.3), 0.05]
            ang = [random.uniform(-math.pi/8, math.pi/8),
                   random.uniform(-math.pi/8, math.pi/8),
                   random.uniform(0, 2*math.pi)]
            quat = p.getQuaternionFromEuler(ang)
            p.resetBasePositionAndOrientation(body_id, pos, quat)
            color = [0, random.uniform(0.3,0.8), 0, random.uniform(0.5,0.9)]
            p.changeVisualShape(body_id, -1, rgbaColor=color)

    def render(self, width=640, height=640, fov_range=(40,70), up_noise=0.1):
        self.change_plane_texture()
        self.randomize_objects_pose()
        fov = random.uniform(*fov_range)
        aspect = width/height
        theta = random.uniform(0, 2*math.pi)
        phi = random.uniform(math.radians(30), math.radians(65))
        radius = random.uniform(0.8, 1.5)
        cam_x = radius * math.cos(phi)*math.cos(theta)
        cam_y = radius * math.cos(phi)*math.sin(theta)
        cam_z = radius * math.sin(phi) + random.uniform(-up_noise, up_noise)
        view = p.computeViewMatrix([cam_x,cam_y,cam_z], [0,0,0.05], [0,0,1])
        proj = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=0.1, farVal=50)
        _, _, rgb, _, seg = p.getCameraImage(width=width, height=height,
                                             viewMatrix=view,
                                             projectionMatrix=proj,
                                             renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.array(rgb)[:, :, :3][:, :, ::-1]
        seg = np.array(seg, dtype=np.int32)
        # build class mask: initialize 0 background
        mask = np.zeros_like(seg, dtype=np.uint8)
        for body_id, class_id in self.objects:
            mask[seg == body_id] = class_id
        return rgb, mask

    def close(self):
        p.disconnect(self.server_id)

if __name__ == '__main__':
    with open(os.path.join('..','config.yml'),'r') as f:
        cfg = yaml.safe_load(f)
    root = os.path.expanduser(cfg['pybullet_dataset'])
    soil_dir = os.path.join(root,'input','soil_resized')
    output = os.path.join(root,'output')
    mesh_dir = os.path.join(root,'input','weedbot_simulation','STLs')

    plants = ['big_plant','small_plant','polygonum_v2','cirsium']
    scale_ranges = {
        'big_plant':    (0.8,1.2),
        'small_plant':  (0.5,0.8),
        'polygonum_v2': (0.008, 0.02),
        'cirsium':      (0.7,1.1)
    }
    max_reps = 4
    samples = 400
    os.makedirs(os.path.join(output,'images'),exist_ok=True)
    os.makedirs(os.path.join(output,'masks'),exist_ok=True)

    world = World(soil_directory=soil_dir)
    for i in range(samples):
        items = []  # will hold mixed objects
        # choose random number of classes to include
        n_classes = random.randint(1, len(plants))
        chosen = random.sample(plants, n_classes)
        for plant in chosen:
            min_s,max_s = scale_ranges[plant]
            reps = random.randint(1, max_reps)
            for _ in range(reps):
                items.append({
                    'mesh_path': os.path.join(mesh_dir,f'{plant}.STL'),
                    'scale': random.uniform(min_s,max_s),
                    'class_id': plants.index(plant)+1
                })
        world.spawn_mixed_objects(items)
        rgb, mask = world.render()
        Image.fromarray(rgb).save(os.path.join(output,'images',f'{i:04d}.png'))
        Image.fromarray(mask,mode='L').save(os.path.join(output,'masks',f'{i:04d}.png'))
    world.close()
