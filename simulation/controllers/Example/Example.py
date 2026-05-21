import os
os.chdir("../../..")

"""ExperimentSupervisor controller."""
from realm_tools.robot_lib.hambot import HamBot


maze_file = 'simulation/worlds/mazes/Samples/Calibration.xml'

# create the robot/supervisor instance.
robot = HamBot()

# Loads the environment from the maze file
robot.load_environment(maze_file)

# Show basic robot/supervisor functions
robot.move_to_random_experiment_start()

robot.move_forward(distance=.5)
robot.move_forward(distance=1)
robot.move_forward(distance=1)
robot.rotate(90)
robot.move_forward(distance=.5)
robot.rotate(90)
robot.move_forward(distance=.5)
robot.move_forward(distance=1)
robot.move_forward(distance=1)
robot.rotate(90)
robot.move_forward(distance=.5)
robot.rotate(90)
robot.move_forward(distance=.5)
robot.move_forward(distance=1)
robot.move_forward(distance=1)

robot.experiment_supervisor.simulationReset()


