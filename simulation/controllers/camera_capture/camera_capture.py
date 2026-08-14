import os
os.chdir("../../..")
print(os.getcwd())

import random
import math

import pandas as pd
from PIL import Image
from tqdm import tqdm

from realm_tools.robot_lib.my_robot import MyRobot


MAZE = 'circ_lm8'
MAZE_FILE = f'simulation/worlds/environments/vpce/{MAZE}.xml'
POSITIONS_FILE = f'simulation/worlds/environments/vpce/positions/{MAZE}_positions.csv'
OUTPUT_DIR = 'data_cache/camera_capture'

N_SAMPLES = 10


os.makedirs(OUTPUT_DIR, exist_ok=True)

robot = MyRobot(enable_cnn_features=False)
robot.load_environment(MAZE_FILE, floor_texture='carpet')

positions = pd.read_csv(POSITIONS_FILE)
sampled = positions.sample(n=N_SAMPLES, replace=False).reset_index(drop=True)

for i, pos in tqdm(sampled.iterrows(), total=len(sampled), desc='camera_capture'):
    theta = random.uniform(0.0, 2 * math.pi)
    robot.teleport_robot(x=pos.x, y=pos.y, theta=theta)
    images, _, _ = robot.capture_pov_images([theta])
    fname = f'camera_{i:02d}.png'
    Image.fromarray(images[0]).save(os.path.join(OUTPUT_DIR, fname))

robot.experiment_supervisor.simulationReset()
