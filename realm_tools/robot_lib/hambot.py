from realm_tools.simulation_lib.environment import Maze
from controller import Supervisor
from matplotlib import patches
import math
import operator

class HamBot(Supervisor):

    # Initiilize an instance of HamBot
    def __init__(self):

        # Inherent from Webots Robot Class: https://cyberbotics.com/doc/reference/robot
        self.experiment_supervisor = Supervisor()

        # Add a display to plot the place cells as they are generated
        self.robot_display = self.experiment_supervisor.getDevice('Robot Display')
        self.robot_display.setOpacity(1.0)

        # Sets Supervisor Root Nodes
        self.root_node = self.experiment_supervisor.getRoot()
        self.children_field = self.root_node.getField('children')
        self.robot_node = self.experiment_supervisor.getSelf()
        self.robot_translation_field = self.robot_node.getField('translation')
        self.robot_rotation_field = self.robot_node.getField('rotation')

        # Physical Robot Specifications
        self.wheel_radius = .045  # meters
        self.axel_length = .205  # meters
        self.robot_radius = .3086  # m

        # Define all systems and makes them class attributes
        self.timestep = int(self.experiment_supervisor.getBasicTimeStep())

        # Webots Rotational Motors: https://cyberbotics.com/doc/reference/motor
        self.left_motor = self.experiment_supervisor.getDevice('left motor')
        self.right_motor = self.experiment_supervisor.getDevice('right motor')
        self.all_motors = [self.left_motor, self.right_motor]
        self.max_motor_velocity = self.left_motor.getMaxVelocity()

        # Initialize robot's motors
        for motor in self.all_motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)

        # Webots Wheel Positional Sensors: https://www.cyberbotics.com/doc/reference/positionsensor
        self.left_encoder = self.experiment_supervisor.getDevice('left wheel encoder')
        self.right_encoder = self.experiment_supervisor.getDevice('right wheel encoder')
        self.all_encoders = [self.left_encoder, self.right_encoder]

        # Initialize robot's encoders
        for encoder in self.all_encoders:
            encoder.enable(self.timestep)


        # Webots Camera: https://cyberbotics.com/doc/reference/camera
        self.camera = self.experiment_supervisor.getDevice('camera')
        self.camera.enable(self.timestep)
        self.camera.recognitionEnable(self.timestep)

        # Webots RpLidarA2: https://www.cyberbotics.com/doc/guide/lidar-sensors#slamtec-rplidar-a2
        self.lidar = self.experiment_supervisor.getDevice('lidar')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        # Webots IMU: https://www.cyberbotics.com/doc/guide/imu-sensors#mpu-9250
        # Webots IMU Accelerometer: https://www.cyberbotics.com/doc/reference/accelerometer
        self.imu = self.experiment_supervisor.getDevice('imu')
        self.imu.enable(self.timestep)

        # Webots GPS:
        self.gps = self.experiment_supervisor.getDevice('gps')
        self.gps.enable(self.timestep)

        self.sensor_calibration()
        self.update_robot_display()

    # Preforms one timestep to update all sensors should be used when initializing robot and after teleport
    def sensor_calibration(self,stop_num =1):
        counter = 0
        while self.experiment_supervisor.step(self.timestep) != -1:
            counter += 1
            if counter > stop_num:
                break

    def get_robot_pose(self):
        while self.experiment_supervisor.step(self.timestep) != -1:
            current_x, current_y, current_z = self.robot_translation_field.getSFVec3f()
            break
        return current_x, current_y, self.get_compass_reading()

    # Reads the robot's IMU compass and return bearing in degrees
    #   North   -> 90
    #   East    -> 0 or 360
    #   South   -> 270
    #   West    -> 180
    def get_compass_reading(self):
        compass_reading = self.imu.getRollPitchYaw()
        bearing = math.degrees(compass_reading[-1])
        if bearing < 0.0:
            bearing += 360.0
        return round(bearing)

    def get_min_lidar_reading(self):
        self.sensor_calibration()
        return min(self.lidar.getRangeImage())

    # Reads current encoder readings and return an array of encoder positions:
    #   [left, right]
    def get_encoder_readings(self):
        return [readings.getValue() for readings in self.all_encoders]

    def sat(self, velocity):
        return max(-self.max_motor_velocity, min(self.max_motor_velocity, velocity))

    # Sets Front Left Motor Velocity (rad/sec)
    def set_left_motor_velocity(self, velocity):
        self.left_motor.setVelocity(self.sat(velocity))

    # Sets Front Right Motor Velocity (rad/sec)
    def set_right_motor_velocity(self, velocity):
        self.right_motor.setVelocity(self.sat(velocity))

    # Sets all motors speed to 0
    def stop(self):
        for motor in self.all_motors:
            motor.setVelocity(0)

    def slow_stop(self):
        while self.experiment_supervisor.step(self.timestep) != -1:
            current_velocity = self.left_motor.getVelocity()
            if current_velocity > .1:
                for motor in self.all_motors:
                    motor.setVelocity(current_velocity / 4)
            else:
                self.stop()
                break

    # Sets all motors speed to velocity
    def go_forward(self, velocity=1):
        for motor in self.all_motors:
            motor.setVelocity(self.sat(velocity))

    # Gets Current Front Left Motor Encoder Reading (meters)
    def get_left_motor_encoder_reading(self):
        return self.left_encoder.getValue()

    # Gets Current Front Right Motor Encoder Reading (meters)
    def get_right_motor_encoder_reading(self):
        return self.right_encoder.getValue()

    # Get Lidar Range Image (meters)
    def get_lidar_range_image(self):
        return self.lidar.getRangeImage()

    def rotate(self, degrees = 90,  Kp=1, Ki=0, Kd=0, margin_error = 0.01):
        I = 0.0
        prev_error = 0.0
        dt = 0.032

        # degrees > 0 -> ccw
        # degrees < 0 -> cw

        setpoint = (self.get_compass_reading() + degrees) % 360

        while self.experiment_supervisor.step(self.timestep) != -1:
            current_heading =  self.get_compass_reading()
            error = (setpoint - current_heading+180)%360 -180 # (-180,180)
            P = Kp * error

            I = Ki * (I + error * dt)

            D = Kd * ((error - prev_error) / dt)

            out_signal = abs(self.sat(P + I + D))
            if -margin_error <= error <= margin_error:
                self.stop()
                break
            elif error < 0:
                self.set_right_motor_velocity(-out_signal)
                self.set_left_motor_velocity(out_signal)
            elif error > 0:
                self.set_right_motor_velocity(out_signal)
                self.set_left_motor_velocity(-out_signal)

            prev_error = error
    def calculate_wheel_distance_traveled(self, starting_encoder_position):
        current_encoder_readings = self.get_encoder_readings()
        differences = list(map(operator.sub, current_encoder_readings, starting_encoder_position))
        average_differences = sum(differences) / len(differences)
        average_distance = average_differences * self.wheel_radius
        return average_distance
    def move_forward(self, distance, Kp=20, margin_error=0.01):
        starting_encoder_position = self.get_encoder_readings()
        while self.experiment_supervisor.step(self.timestep) != -1:
            error = distance - self.calculate_wheel_distance_traveled(starting_encoder_position)
            if error <= margin_error:
                self.stop()
                break
            self.go_forward(velocity=self.sat(Kp * error))


    # Supervisor Functions: allows robot to control the simulation
    # DO NOT MODIFY: unless you are attempting to manipulate the webots world simulations!!!

    # Takes in a xml maze file and creates the walls, starting locations, and goal locations
    def load_environment(self, maze_file,display =False):
        self.maze = Maze(maze_file, display_width=self.robot_display.getWidth(),
                         display_height=self.robot_display.getHeight())
        if display:
            self.maze_figure, self.maze_figure_ax = self.maze.get_maze_figure()
            self.maze_figure.savefig('data/DataCache/maze.png')
            self.update_robot_display(name='maze')

        self.obstical_nodes = []
        self.boundry_wall_nodes = []
        self.cylinder_landmark_nodes = []
        self.tag_landmark_nodes = []

        for obstacles in self.maze.obstacles:
            self.children_field.importMFNodeFromString(-1, obstacles.get_webots_node_string())
            self.obstical_nodes.append(self.experiment_supervisor.getFromDef('Obstacle'))
        for boundary_wall in self.maze.boundary_walls:
            self.children_field.importMFNodeFromString(-1, boundary_wall.get_webots_node_string())
            self.boundry_wall_nodes.append(self.experiment_supervisor.getFromDef('Obstacle'))
        for cylinder_landmark in self.maze.cylinder_landmarks:
            self.children_field.importMFNodeFromString(-1, cylinder_landmark.get_webots_node_string())
            self.cylinder_landmark_nodes.append(self.experiment_supervisor.getFromDef('Landmark'))
        for tag_landmark in self.maze.tag_landmarks:
            self.children_field.importMFNodeFromString(-1, tag_landmark.get_webots_node_string())
            self.tag_landmark_nodes.append(self.experiment_supervisor.getFromDef('RectangularPanel'))

    def reset_environment(self):
        self.teleport_robot(theta=math.pi / 2)
        total_nodes = len(self.obstical_nodes) + len(self.boundry_wall_nodes) + len(self.landmark_nodes)
        for i in range(total_nodes):
            self.children_field.removeMF(-1)
        self.maze.close_maze_figure()
        self.sensor_calibration()

    # Teleports the robot to the point (x,y,z)
    def teleport_robot(self, x=0.0, y=0.0, z=0.09, theta=math.pi):
        self.robot_translation_field.setSFVec3f([x, y, z])
        self.robot_rotation_field.setSFRotation([0, 0, 1, theta])
        self.sensor_calibration()

    def move_to_training_start(self):
        starting_position = self.maze.experiment_starting_location[0]
        self.teleport_robot(starting_position.x, starting_position.y, theta=starting_position.theta)

    # Moves the robot to a random starting position
    def move_to_testing_start(self, index=-1):
        if index == -1:
            starting_position = self.maze.get_random_experiment_testing_starting_position()
        else:
            starting_position = self.maze.experiment_starting_location[index]
        self.teleport_robot(starting_position.x, starting_position.y, theta=starting_position.theta)
        return self.get_robot_pose()

    # Moves the robot to a random starting position
    def move_to_random_experiment_start(self):
        starting_position = self.maze.get_random_experiment_starting_position()
        self.teleport_robot(starting_position.x, starting_position.y, theta=starting_position.theta)
        return self.get_robot_pose()

    # Moves the robot to a random starting position
    def move_to_habituation_start(self, index=-1):
        if index == -1:
            starting_position = self.maze.get_random_habituation_starting_position()
        else:
            starting_position = self.maze.habituation_start_location[index]
        self.teleport_robot(starting_position.x, starting_position.y, theta=starting_position.theta)
        return self.get_robot_pose()

    def check_at_goal(self):
        current_x, current_y, currentcurrent_z = self.robot_translation_field.getSFVec3f()
        goal_x, goal_y = self.maze.get_goal_location()
        distance_to_goal = math.sqrt((current_x - goal_x) ** 2 + (current_y - goal_y) ** 2)
        return distance_to_goal < 0.8

    def get_dist_to_goal(self):
        current_x, current_y, currentcurrent_z = self.robot_translation_field.getSFVec3f()
        goal_x, goal_y = self.maze.get_goal_location()
        distance_to_goal = math.sqrt((current_x - goal_x) ** 2 + (current_y - goal_y) ** 2)
        return distance_to_goal

    # Plots Place cells and shows them on the Display
    def update_robot_display(self,name='default'):
        while self.experiment_supervisor.step(self.timestep) != -1:
            ir = self.robot_display.imageLoad('data/DataCache/'+name+'.png')
            self.robot_display.imagePaste(ir, 0, 0, True)
            break
