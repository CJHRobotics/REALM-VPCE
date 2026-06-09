import os
os.chdir("../../..")
print(os.getcwd())

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm

from realm_tools.robot_lib.my_robot import MyRobot
from realm_tools.experiment_lib.loggers.visual_data_set import PovDataset
from realm_tools.image_lib.image_feature_lib import extract_combined_features


maze_file_dir = 'simulation/worlds/environments/vpce/'

# maze_files = ['lm8', 'lm8_r45', 'lm8_r90','lm8_o6']
maze_files = ['four_room']


# How many positions to capture before extracting features.
# Larger = more parallelism, more RAM. Tune to your machine.
# At ~200KB per image and 8 headings: 50 positions ≈ 80MB, 200 positions ≈ 320MB.
BATCH_SIZE = 500

# Feature extraction flags — set to False to exclude a descriptor.
# The output vector always follows the order: colour hist · spatial.
USE_HOG        = True
USE_COLOR_HIST = True
USE_SPATIAL    = True

# Heading aggregation flag.
# True  → average the 8 heading feature vectors into one  (output dim = feature_dim)
# False → concatenate all 8 heading vectors               (output dim = 8 × feature_dim)
USE_AVERAGE    = True

thetas = [0.0, 0.7854, 1.5708, 2.3562, 3.1416, 3.9270, 4.7124, 5.4978]

# Build the extractor with the configured flags baked in
_extractor = partial(extract_combined_features,
                     use_hog=USE_HOG,
                     use_color_hist=USE_COLOR_HIST,
                     use_spatial=USE_SPATIAL)

robot = MyRobot(enable_cnn_features=False, cnn_extractor_model='mobilenetv3')


def flush_batch(batch, dataset):
    """Extract features for all images in the batch in parallel."""
    all_images = [img for images, *_ in batch for img in images]
    with ThreadPoolExecutor() as executor:
        all_features = list(executor.map(_extractor, all_images))
    n = len(thetas)
    for i, (_, x, y, theta) in enumerate(batch):
        window = all_features[i * n : (i + 1) * n]   # n feature vectors for this position
        if USE_AVERAGE:
            multimodal = np.mean(window, axis=0)       # (feature_dim,)
        else:
            multimodal = np.concatenate(window)        # (8 × feature_dim,)
        dataset.add_observations(multimodal, None, x, y, theta)


for maze_index, maze in enumerate(maze_files):
    print(f"\n{'='*50}")
    print(f"Collecting data: {maze}  ({maze_index + 1}/{len(maze_files)})")
    print(f"{'='*50}")

    # Load (or reload) the environment for this maze
    if maze_index == 0:
        robot.load_environment(maze_file_dir + maze + '.xml', floor_texture='carpet')
    else:
        robot.reset_environment()
        robot.load_environment(maze_file_dir + maze + '.xml', floor_texture='carpet')

    positions_path = maze_file_dir + 'positions/' + maze + '_positions.csv'
    # positions_path = maze_file_dir + 'positions/lm8_positions.csv'
    positions = pd.read_csv(positions_path)

    dataset = PovDataset()
    batch   = []

    with tqdm(total=len(positions), desc=maze) as pbar:
        for _, pos in positions.iterrows():
            robot.teleport_robot(x=pos.x, y=pos.y, theta=pos.theta)
            images = robot.capture_pov_images(thetas)
            batch.append((images, pos.x, pos.y, pos.theta))

            if len(batch) >= BATCH_SIZE:
                flush_batch(batch, dataset)
                batch = []

            pbar.update(1)

    if batch:
        flush_batch(batch, dataset)

    dataset.save_dataset('data/vpce/collect_data/' + maze)
    print(f"Saved: data/vpce/collect_data/{maze}")

robot.experiment_supervisor.simulationReset()
