import os
os.chdir("../../..")
print(os.getcwd())

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from realm_tools.robot_lib.my_robot import MyRobot
from realm_tools.experiment_lib.loggers.visual_data_set import PovDataset
from realm_tools.image_lib.image_feature_lib import extract_combined_features


maze_file_dir = 'simulation/worlds/environments/vpce/'

maze_files = ['lm8', 'lm8_r45', 'lm8_r90', 'lmo8']
maze_index  = 2

# How many positions to capture before extracting features.
# Larger = more parallelism, more RAM. Tune to your machine.
# At ~200KB per image and 8 headings: 50 positions ≈ 80MB, 200 positions ≈ 320MB.
BATCH_SIZE = 500

thetas = [0.0, 0.7854, 1.5708, 2.3562, 3.1416, 3.9270, 4.7124, 5.4978]

robot = MyRobot(enable_cnn_features=False, cnn_extractor_model='mobilenetv3')
robot.load_environment(maze_file_dir + maze_files[maze_index] + '.xml',floor_texture="carpet")

# positions_path = maze_file_dir + 'positions/' + maze_files[maze_index] + '_positions.csv'
positions_path = maze_file_dir + 'positions/lm8_positions.csv'
positions = pd.read_csv(positions_path)

dataset = PovDataset()

# Accumulate (images, pose) tuples across BATCH_SIZE positions then
# extract features for the whole batch in parallel.
batch = []   # list of (images: list[ndarray], x, y, theta)

def flush_batch(batch):
    """Extract features for all images in the batch in parallel."""
    all_images = [img for images, *_ in batch for img in images]
    with ThreadPoolExecutor() as executor:
        all_features = list(executor.map(extract_combined_features, all_images))
    n = len(thetas)
    for i, (_, x, y, theta) in enumerate(batch):
        multimodal = np.concatenate(all_features[i * n : (i + 1) * n])
        dataset.add_observations(multimodal, None, x, y, theta)

with tqdm(total=len(positions), desc="Collecting data") as pbar:
    for i, (_, pos) in enumerate(positions.iterrows()):
        robot.teleport_robot(x=pos.x, y=pos.y, theta=pos.theta)
        images = robot.capture_pov_images(thetas)
        batch.append((images, pos.x, pos.y, pos.theta))

        if len(batch) >= BATCH_SIZE:
            flush_batch(batch)
            batch = []

        pbar.update(1)

# Process any remaining positions
if batch:
    flush_batch(batch)

dataset.save_dataset('data/vpce/collect_data/' + maze_files[maze_index])
robot.experiment_supervisor.simulationReset()
