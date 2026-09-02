import os
os.chdir("../../..")
print(os.getcwd())

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm

from realm_tools.robot_lib.my_robot import MyRobot
from realm_tools.experiment_lib.loggers.visual_data_set import PovDataset
from realm_tools.image_lib.image_feature_lib import extract_feature_dict


maze_file_dir = 'simulation/worlds/environments/vpce/'

# Environment-size sweep (experiment 3). Circular arenas of increasing area
# with landmarks held at a fixed 0.75 m, replicating Harland's fixed room
# cues. Every grid carries ~30,100 positions regardless of area, so sample
# count is not a variable -- at a constant 0.1 m spacing these would have
# ranged 1,018 to 30,172 and field count would have scaled with area whether
# or not the model did anything. circ_lm8_r0 is the r = 10 member and is
# already collected.
maze_files = ['circ_lm8_rad1p25', 'circ_lm8_rad2p0',
              'circ_lm8_rad3p5', 'circ_lm8_rad6p0']

# Overridable so one arena can be collected per SLURM job and the sweep run
# in parallel, instead of serially inside a single long Webots session.
if os.environ.get('REALM_MAZES'):
    maze_files = [m.strip() for m in os.environ['REALM_MAZES'].split(',') if m.strip()]
    print(f'REALM_MAZES override: {maze_files}')

# Already collected, add back to regenerate:
#   landmark-count sweep  circ_lm4_r0, circ_lm6_r0, circ_lm8_r0, circ_lm10_r0
#   geometry sweep        rect_lm8_r0, corr_lm8_r0

# Positions file per maze. Falls back to this literal name if the per-maze
# CSV is missing (matches the older circular-arena convention). The
# circ_lm* worlds have no per-maze CSV of their own and land here.
POSITIONS_FILE_FALLBACK = 'circ_lm8_r0_positions.csv'



# How many positions to capture before extracting features.
# Larger = more parallelism, more RAM. Tune to your machine.
# At ~200KB per image and 8 headings: 50 positions ≈ 80MB, 200 positions ≈ 320MB.
BATCH_SIZE = 500

# Feature extraction flags — set to False to exclude a descriptor.
# The output vector always follows the order: colour hist · spatial.
USE_HOG        = True
USE_COLOR_HIST = True
USE_SPATIAL    = True

thetas = [0.0, 0.7854, 1.5708, 2.3562, 3.1416, 3.9270, 4.7124, 5.4978]

# Build the extractor with the configured flags baked in. Individual
# feature blocks are saved as their own HDF5 fields so downstream code
# can freely mix and match (hog only, colour only, hog + lidar, ...).
_extractor = partial(extract_feature_dict,
                     use_hog=USE_HOG,
                     use_color_hist=USE_COLOR_HIST,
                     use_spatial=USE_SPATIAL)

robot = MyRobot(enable_cnn_features=False, cnn_extractor_model='mobilenetv3')


def flush_batch(images, meta, dataset):
    """Extract features for all images in the batch in parallel and log each as an observation.

    Each descriptor block (hog / color_hist / spatial) is saved as its own
    per-heading field. Lidar is a 360-degree scan that already covers all
    directions from one reading, so we log it *once per position* (only
    when lidar_scan is not None in the meta tuple) instead of broadcasting
    the same scan to every heading row.
    """
    with ThreadPoolExecutor() as executor:
        feature_dicts = list(executor.map(_extractor, images))
    for (x, y, theta, landmark_mask, landmark_azimuths, lidar_scan), fd in zip(meta, feature_dicts):
        kwargs = dict(x=x, y=y, theta=theta,
                       landmark_mask=landmark_mask,
                       landmark_azimuths=landmark_azimuths,
                       **fd)
        if lidar_scan is not None:
            kwargs['lidar'] = lidar_scan
        dataset.add_observation(**kwargs)


# True once an environment has been loaded into Webots. Tracked separately
# from the loop index because a maze can be skipped as already-collected,
# and reset_environment() is only valid after something has been loaded.
env_loaded = False

for maze_index, maze in enumerate(maze_files):
    print(f"\n{'='*50}")
    print(f"Collecting data: {maze}  ({maze_index + 1}/{len(maze_files)})")
    print(f"{'='*50}")

    out_path = 'data/vpce/collect_data/' + maze
    if os.path.exists(out_path + '.h5'):
        print(f"Already collected, skipping: {out_path}.h5")
        continue

    # Load (or reload) the environment for this maze
    if env_loaded:
        robot.reset_environment()
    robot.load_environment(maze_file_dir + maze + '.xml', floor_texture='carpet')
    env_loaded = True

    positions_path = maze_file_dir + 'positions/' + maze + '_positions.csv'
    if not os.path.exists(positions_path):
        positions_path = maze_file_dir + 'positions/' + POSITIONS_FILE_FALLBACK
    positions = pd.read_csv(positions_path)
    print(f"Positions: {positions_path}   ({len(positions)} points)")

    dataset = PovDataset()
    batch_images = []
    batch_meta   = []  # (x, y, theta, landmark_mask, landmark_azimuths) per image, parallel to batch_images

    with tqdm(total=len(positions), desc=maze) as pbar:
        for _, pos in positions.iterrows():
            robot.teleport_robot(x=pos.x, y=pos.y, theta=pos.theta)
            images, landmark_masks, landmark_azimuths, lidar_scan = robot.capture_pov_images(thetas)
            batch_images.extend(images)
            # Attach the lidar scan only to the first (north-facing) heading
            # row per position; every other row carries None so the lidar
            # field ends up with exactly one entry per location.
            batch_meta.extend(
                (pos.x, pos.y, theta, mask, azimuths,
                 lidar_scan if i == 0 else None)
                for i, (theta, mask, azimuths) in enumerate(
                    zip(thetas, landmark_masks, landmark_azimuths))
            )

            if len(batch_meta) >= BATCH_SIZE * len(thetas):
                flush_batch(batch_images, batch_meta, dataset)
                batch_images, batch_meta = [], []

            pbar.update(1)

    if batch_images:
        flush_batch(batch_images, batch_meta, dataset)

    dataset.save_dataset(out_path)
    print(f"Saved: {out_path}")

# Quit rather than reset. simulationReset() restarts the controllers, which
# unattended means this script runs again, skips every already-collected
# maze, and resets once more -- forever. Quitting also lets a batch job
# finish instead of sitting until its walltime expires.
robot.experiment_supervisor.simulationQuit(0)
