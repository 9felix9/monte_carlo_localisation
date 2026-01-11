import rclpy
from rclpy.node import Node
from mcl import MCL
from landmark_manager import LandmarkManager

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
from geometry_msgs.msg import Quaternion

class MCLNode(Node):
    def __init__(self):
        super().__init__('mcl_node')
        self.get_logger().info("Monte Carlo Localization node started")

        self.last_time = None
        self.landmark_manager = LandmarkManager()
        if not self.landmark_manager.load_from_csv("src/kalman_positioning/landmarks.csv"):
            self.get_logger().error("Could not load landmarks!")
    
        self.mcl = MCL()
        if not self.mcl: 
            self.get_logger().error("Could not load the MCL class!")

        ## read in the landmarks.csv - could be done by subscribing to landmarks_gt topic aswell
        lm = LandmarkManager()
        lm.load_from_csv("/home/felix/Schreibtisch/projects/robotics_hw2/src/mcl_localization/landmarks.csv") # LMs saved in lm obj

        # get min and max from x and y
        self.mcl.landmarks_gt = lm.get_all_landmarks()
    

        ### Algo start ###

        self.mcl.initializeParticles()


        # ROS Subscribers
        # -----------------------------------------------------------
        self.create_subscription(Odometry, "/robot_noisy", self.odometry_callback, 10)
        self.create_subscription(PointCloud2, "/landmarks_observed", self.landmark_callback, 10)
        self.create_subscription(Odometry, "/robot_gt", self.gt_callback, 10)

    def odometry_callback(self, msg: Odometry): 
        # each 10ms this callback gets executed getting the odometry data from the fake robot
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if not self.last_time: 
            self.last_time = t

        # read out motion from the fake robot odometry ( velocity model)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vtheta = msg.twist.twist.angular.z # angular velocity

        # time diff
        dt = t - self.last_time

        self.mcl.motionUpdate(vx, vy, vtheta, dt)

    def landmark_callback(self, msg: PointCloud2): 
        self.mcl.landmarks_observed = msg
        test = 1

        for p in read_points(
            msg,
            field_names=("x", "y", "id"),
            skip_nans=True
        ):
            obs_x_r, obs_y_r, lm_id = p
            self.mcl.landmarks_observed[lm_id] = (obs_x_r, obs_y_r)

        self.mcl.measurementUpdate()

        
    
def main():
    rclpy.init()
    node = MCLNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
