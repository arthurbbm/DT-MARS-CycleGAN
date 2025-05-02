import math
import random
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import re
import time
from glob import glob
import pandas as pd
import os
from scipy import ndimage
import yaml


def mask_find_bboxs(mask):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stats = stats[stats[:, 4].argsort()]
    return stats

def mask_find_multi_bboxs(mask):
    objects = ndimage.find_objects(mask)
    return objects


class World(object):
    def __init__(self, object_mesh_path, soil_directory):
        self.soil_dir = soil_directory
        # self.server_id = p.connect(p.GUI)
        self.server_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.plane_id = p.loadURDF("plane.urdf")
        base_position = [0,0,0.05]
        # base_position = [0, 0, -40]

        """
        change the object urdf file, choose from below

        ['big_plant', 'big_plant_src', 'cirsium', 'cirsium_v1',
        'crabgrass_v1', 'crabgrass_v2', 'polygonum_v1', 'polygonum_v2', 'small_plant']

        bbox not good: ['CottonPlant_v1', 'CottonPlant_v2', 'CottonPlant_v3']

        To adjust the plant object size, go to urdf file and change 'scale'
        """
        
        # self.object_id = p.loadURDF('./weedbot_simulation/weedbot_description/urdf/weedbot.urdf', basePosition=base_position) # weedbot robot car
        # self.object_id = p.loadURDF('./weedbot_simulation/simulation_world/urdf/{}.urdf'.format(plant), basePosition=base_position) # weeds
        assert os.path.exists(object_mesh_path), "File does not exist: {}".format(object_mesh_path)
        self.object_id = self.load_plant_mesh(object_mesh_path, base_position, scale=[0.04, 0.04, 0.04], mass=0.0)
        self.rotate_object(euler_angles=[0, 0, 0])

        # multi objs
        # self.object_ids = []
        # for _ in range(3):
        #     object_id = p.loadURDF('./weedbot_simulation/simulation_world/urdf/{}.urdf'.format(plant), basePosition=base_position)
        #     self.object_ids.append(object_id)

    def load_plant_mesh(self, mesh_path, base_position, scale=[1, 1, 1], mass=0.0):
        """
        Loads an arbitrary STL mesh into the simulation with separate
        visual and (convex hull) collision shapes.

        :param mesh_path:    Path to the .stl or .obj mesh file
        :param base_position: [x,y,z] world position
        :param scale:        [sx,sy,sz] mesh scaling factors
        :param mass:         Mass of the object (0 = static)
        :returns:            body unique ID
        """
        # Create convex collision shape
        col_shape = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=mesh_path,
            meshScale=scale
        )
        # Create full-detail visual shape
        vis_shape = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=mesh_path,
            meshScale=scale
        )
        # Combine into a single rigid body
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=base_position
        )
        return body_id

    def rotate_object(self, euler_angles=None):
        """
        Rotates the loaded object. If euler_angles is None, applies a random yaw.
        :param euler_angles: [roll, pitch, yaw] in radians
        """
        # Get current position
        pos, _ = p.getBasePositionAndOrientation(self.object_id)
        # Determine rotation
        if euler_angles is None:
            yaw = random.uniform(0, 2 * math.pi)
            euler_angles = [0, 0, yaw]
        quat = p.getQuaternionFromEuler(euler_angles)
        # Reset orientation while keeping position
        p.resetBasePositionAndOrientation(self.object_id, pos, quat)

    def close(self):
        p.disconnect(self.server_id)

    def change_plane(self):
        texture_list = glob('{}/*.jpg'.format(self.soil_dir))
        texture_path = np.random.choice(texture_list)
        texture_id = p.loadTexture(texture_path)
        p.changeVisualShape(self.plane_id, -1, textureUniqueId=texture_id)

    def step(self):
        self.change_plane()
        base_position = self.reset_object()
        rgb, seg, gt_bbox = self.get_image(base_position)
        return rgb, seg, gt_bbox

    def reset_object(self):
        base_position = [random.random(),random.random(),0.05]
        # p.resetBasePositionAndOrientation(self.object_id, base_position,[math.pi/2,0,0,1])
        p.resetBasePositionAndOrientation(self.object_id, base_position,[0,0,0,1])
        # change color of object to green like
        alpha = np.random.uniform(0.4, 0.8)
        alpha = 0.9
        green = np.random.uniform(0.4, 0.8)
        coneColor = [0, green, 0, alpha]
        p.changeVisualShape(self.object_id, -1, rgbaColor=coneColor)
    
        # multi objs
        # anchor_position = [random.random(),random.random(),0.05]
        # shift = [[-random.uniform(0.2, 0.3), -random.uniform(0.2, 0.3), 0],
        #          [0, 0, 0],
        #          [random.uniform(0.2, 0.3), random.uniform(0.2, 0.3), 0]]

        # for i, object_id in enumerate(self.object_ids):
        #     base_position = [x+y for x,y in zip(anchor_position,shift[i])]
        #     p.resetBasePositionAndOrientation(object_id, base_position, [0, 0, 0, 1])

        #     # Change color of object to a random shade of green
        #     alpha = np.random.uniform(0.4, 0.8)
        #     green = np.random.uniform(0.4, 0.8)
        #     coneColor = [0, green, 0, alpha]
        #     p.changeVisualShape(object_id, -1, rgbaColor=coneColor)
        return base_position

    def get_image(self, base_position):
        # adjust camera hight and distance here
        r = 0.8 + 0.4*random.random() # distance
        # t = 2 * math.pi * random.random()
        t = 0
        h = 0.8 + 0.4 * random.random() # hight
        camera_pos = [base_position[0] + r*math.sin(t), base_position[1] + r*math.cos(t), base_position[2]+h]

        target_position = [base_position[0]-0.2*random.random()+0.1,
                           base_position[1]-0.2*random.random()+0.1,
                           base_position[2]-0.1*random.random()+0.05]

        view_mat = p.computeViewMatrix(camera_pos, target_position, [0, 0, 1], self.server_id)
        proj_mat = p.computeProjectionMatrixFOV(fov=49.1,
                                                aspect=1.0,
                                                nearVal=0.1,
                                                farVal=100,
                                                physicsClientId=self.server_id)
        w, h, rgb, depth, seg = p.getCameraImage(width=640,
                                                 height=640,
                                                 viewMatrix=view_mat,
                                                 projectionMatrix=proj_mat,
                                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.array(rgb)[:, :, :3][:, :, ::-1]
        rgb = np.array(rgb, dtype=np.uint8)

        seg = np.array(seg, dtype=np.uint8)
        ret, thresh = cv2.threshold(seg, 0, 255, 0)
        bboxes = mask_find_bboxs(thresh)
        gt_bbox = np.zeros((1,4))
        for b in bboxes:
            if (b[0] + b[1]) < 100:
                continue
            gt_bbox[0,0] = b[0]/w
            gt_bbox[0,1] = b[1]/h
            gt_bbox[0,2] = b[0]/w + b[2]/w
            gt_bbox[0,3] = b[1]/h + b[3]/h

        # multi obj
        # bboxes = mask_find_multi_bboxs(seg)
        # gt_bbox = np.zeros((3,4))
        # for i, b in enumerate(bboxes):
        #     sy, sx = b
        #     gt_bbox[i,0] = sx.start/w
        #     gt_bbox[i,1] = sy.start/h
        #     gt_bbox[i,2] = sx.stop/w
        #     gt_bbox[i,3] = sy.stop/h
        return rgb,seg,gt_bbox

if __name__ == '__main__':

    
    plants = ['big_plant', 'small_plant', 'polygonum_v2', 'cirsium'] # 800 800 400 400
    
    with open("../config.yml", "r") as f:
        config = yaml.safe_load(f)

    dataset_path = os.path.expanduser(config["pybullet_dataset"])

    output_dir = os.path.join(dataset_path, "output")
    input_dir = os.path.join(dataset_path, "input")
    plant_mesh_file = os.path.join(input_dir, "weedbot_simulation", "STLs", "cirsium_v1.STL")
    soil_dir = os.path.join(input_dir, "soil_resized")
    env = World(object_mesh_path=plant_mesh_file, soil_directory=soil_dir)
    plant_mesh_file = os.path.splitext(os.path.basename(plant_mesh_file))[0]
    bbox = []
    for i in range(800):
        rgb, seg, gt_bbox = env.step()
        seg[seg>0] = 255
        seg3 = np.stack([seg, seg, seg], axis=2)

        color = (0, 0, 255) # Red color in BGR；红色：rgb(255,0,0)
        thickness = 1 # Line thickness of 1 px
        (h,w,c) = rgb.shape

        centerX = (gt_bbox[0,0] + gt_bbox[0,2]) / 2
        centerY = (gt_bbox[0,1] + gt_bbox[0,3]) / 2
        width = gt_bbox[0,2] - gt_bbox[0,0]
        height = gt_bbox[0,3] - gt_bbox[0,1]
    
        start_point = (int(gt_bbox[0,0]*w), int(gt_bbox[0,1]*h))
        end_point = (int(gt_bbox[0,2]*w), int(gt_bbox[0,3]*h))
        # cv2.rectangle(rgb, start_point, end_point, color, thickness) # add red bbox in image

        # cv2.imshow('show_image', rgb)
        # make sure seg3 is uint8
        seg3 = np.stack([seg, seg, seg], axis=2).astype(np.uint8)

        # stack RGB | SEG
        # vis = np.hstack((rgb, seg3))

        # cv2.imshow('RGB (left) | Seg (right)', vis)
        # if cv2.waitKey(1) & 0xFF == 27:
            # break

        img_path = 'images/{}_{:04d}.jpg'.format(plant_mesh_file, i)
        img_path = os.path.join(output_dir, img_path)
        if not os.path.exists(os.path.dirname(img_path)):
            os.makedirs(os.path.dirname(img_path))
        cv2.imwrite(img_path, rgb)

        label_path = 'labels/{}_{:04d}.txt'.format(plant_mesh_file, i)
        label_path = os.path.join(output_dir, label_path)
        if not os.path.exists(os.path.dirname(label_path)):
            os.makedirs(os.path.dirname(label_path))
        with open(label_path, 'w') as f:
            f.write("0 {} {} {} {}\n".format(centerX, centerY, width, height))

        segmentation_path = 'segmentation/{}_{:04d}.jpg'.format(plant_mesh_file, i)
        segmentation_path = os.path.join(output_dir, segmentation_path)
        if not os.path.exists(os.path.dirname(segmentation_path)):
            os.makedirs(os.path.dirname(segmentation_path))
        cv2.imwrite(segmentation_path, seg3)

        # if cv2.waitKey(1) == 27:
            # break

    # df = pd.DataFrame(bbox)
    # df.to_csv('bbox/{}_bbox.csv'.format(plant), index=False)



