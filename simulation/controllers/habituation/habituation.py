import os
os.chdir("../../..")
print(os.getcwd())
from realm_tools.robot_lib.my_robot import MyRobot
from realm_tools.experiment_lib.loggers.visual_data_set import PovDataset
from tqdm import tqdm


maze_file_dir = 'simulation/worlds/environments/vpce/'


maze_files = ['LM8', 'LM8D', 'LMO8', 'LMO8D']
maze_index = 0

# create the robot/supervisor instance.
robot = MyRobot(enable_cnn_features=False,cnn_extractor_model='mobilenetv3')
robot.load_environment(maze_file_dir + maze_files[maze_index] + '.xml')

dataset = PovDataset()

thetas = [0.0, 0.7854, 1.5708, 2.3562, 3.1416, 3.9270, 4.7124, 5.4978]
total_steps = len(robot.maze.training_starting_location)
with tqdm(total=total_steps, desc="Collecting data") as pbar:
    for start_position in robot.maze.training_starting_location:
        robot.teleport_robot(x=start_position.x, y=start_position.y, theta=start_position.theta)
        robot_x, robot_y, robot_theta = robot.get_robot_pose()
        multimodal_features, cnn_features = robot.get_full_robot_pov_features(thetas)
        dataset.add_observations(multimodal_features, cnn_features, robot_x, robot_y, robot_theta)
        pbar.update(1)

dataset.save_dataset("data/vpce/habituation/" + maze_files[maze_index])

robot.experiment_supervisor.simulationReset()

