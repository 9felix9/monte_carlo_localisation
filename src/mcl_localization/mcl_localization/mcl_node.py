import math
import rclpy
from rclpy.node import Node
from mcl_localization.mcl import MCL
from mcl_localization.landmark_manager import LandmarkManager

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseArray, Pose

class MCLNode(Node):
    def __init__(self):
        super().__init__('mcl_node')
        self.get_logger().info("Monte Carlo Localization node started")

        self.last_time = None
        self.landmark_manager = LandmarkManager()
        if not self.landmark_manager.load_from_csv("/home/felix/Schreibtisch/projects/robotics_hw2/src/mcl_localization/landmarks.csv"):
            self.get_logger().error("Could not load landmarks!")

        self.mcl = MCL()
        if not self.mcl: 
            self.get_logger().error("Could not load the MCL class!")

        # get min and max from x and y
        self.mcl.landmarks_gt = self.landmark_manager.get_all_landmarks()
    

        ### Algo start ###

        self.mcl.initializeParticles()


        # ROS Subscribers
        # -----------------------------------------------------------
        self.create_subscription(Odometry, "/robot_noisy", self.odometry_callback, 10)
        self.create_subscription(PointCloud2, "/landmarks_observed", self.landmark_callback, 10)
        # self.create_subscription(Odometry, "/robot_gt", self.gt_callback, 10)

        # ROS Publisher
        self.estimated_pub = self.create_publisher(Odometry, "/robot_estimated_odometry", 10)
        self.particles_pub = self.create_publisher(PoseArray, "/mcl/particles",10)
    
    
    def odometry_callback(self, msg: Odometry): 
        # each 10ms this callback gets executed getting the odometry data from the fake robot
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_time is None: 
            self.last_time = t
            return
        
        # read out motion from the fake robot odometry ( velocity model)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vtheta = msg.twist.twist.angular.z # angular velocity

        # time diff
        dt = t - self.last_time

        self.mcl.motionUpdate(vx, vy, vtheta, dt)
        self.last_time = t

    def landmark_callback(self, msg: PointCloud2): 
        # self.mcl.landmarks_observed = msg
        self.mcl.landmarks_observed = {}

        for p in read_points(
            msg,
            field_names=("x", "y", "id"),
            skip_nans=True
        ):
            obs_x_r, obs_y_r, lm_id = p
            self.mcl.landmarks_observed[int(lm_id)] = (obs_x_r, obs_y_r)

        self.mcl.measurementUpdate()
        now = self.get_clock().now().to_msg()
        self.publish_estimated_pose(now)
        self.publish_particles(now)


    def publish_estimated_pose(self, timestamp): 
        # estimated_pose = [x, y, theta]
        estimated_pose = self.mcl.estimatePose()

        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"

        # Position
        odom.pose.pose.position.x = float(estimated_pose[0])
        odom.pose.pose.position.y = float(estimated_pose[1])
        odom.pose.pose.position.z = 0.0

        # Orientation: theta -> quaternion
        theta = float(estimated_pose[2])

        # method needs roll, pitch and yaw
        # roll: kippen links rechts
        # pitch: nase hoch runter 
        # roll: drehen um die hochachse, also links/rechts drehen

        odom.pose.pose.orientation = self.yaw_to_quaternion(theta)

        # Publish
        self.estimated_pub.publish(odom)

    def publish_particles(self, timestamp):
        msg = PoseArray()
        msg.header.stamp = timestamp
        msg.header.frame_id = "map"

        # self.mcl.Particles is (N, 3): [x, y, theta]
        for x, y, theta in self.mcl.Particles:
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = 0.0

            q = self.yaw_to_quaternion(float(theta))
            pose.orientation = q

            msg.poses.append(pose)

        self.particles_pub.publish(msg)

    @staticmethod
    def yaw_to_quaternion(theta: float) -> Quaternion:
        half = 0.5 * float(theta)
        return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))
    
def main():
    rclpy.init()
    node = MCLNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
