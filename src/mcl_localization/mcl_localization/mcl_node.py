import math
import rclpy
import numpy as np
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

        self.last_odom_pose = None
        self.last_time = None
        self.landmark_manager = LandmarkManager()
        if not self.landmark_manager.load_from_csv("/home/felix/Schreibtisch/projects/robotics_hw2/src/mcl_localization/landmarks.csv"):
            self.get_logger().error("Could not load landmarks!")
        
        # init mcl variables
        self.mcl = MCL(logger=self.get_logger())
        if not self.mcl: 
            self.get_logger().error("Could not load the MCL class!")

        # get min and max from x and y
        self.mcl.landmarks_gt = self.landmark_manager.get_all_landmarks()
    

        ### Algo start ###

        self.mcl.initializeParticles()
        self.log_particles("First init")


        # ROS Subscribers
        # -----------------------------------------------------------
        self.create_subscription(Odometry, "/robot_noisy", self.odometry_model_callback, 10)
        self.create_subscription(PointCloud2, "/landmarks_observed", self.landmark_callback, 10)
        # self.create_subscription(Odometry, "/robot_gt", self.gt_callback, 10)

        # ROS Publisher
        self.estimated_pub = self.create_publisher(Odometry, "/robot_estimated_odometry", 10)
        self.particles_pub = self.create_publisher(PoseArray, "/mcl/particles",10)
    
    
    def velocity_model_callback(self, msg: Odometry): 
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

        self.log_particles("before motion update") 
        self.mcl.motionUpdate(vx, vy, vtheta, dt)
        self.log_particles("after motion update") 
        self.last_time = t

    def odometry_model_callback(self, msg: Odometry):
        # Current odom pose
        x_curr = msg.pose.pose.position.x
        y_curr = msg.pose.pose.position.y
        theta_curr = MCLNode.quat_to_yaw(msg.pose.pose.orientation)

        if self.last_odom_pose is None:
            self.last_odom_pose = (x_curr, y_curr, theta_curr)
            return

        x_prev, y_prev, theta_prev = self.last_odom_pose

        dx = x_curr - x_prev
        dy = y_curr - y_prev

        delta_trans = math.sqrt(dx * dx + dy * dy)

        # If there is (almost) no translation, rot1 is ambiguous; handle robustly
        if delta_trans < 1e-9:
            delta_rot1 = 0.0
        else:
            delta_rot1 = math.atan2(dy, dx) - theta_prev

        delta_rot2 = theta_curr - theta_prev - delta_rot1

        # Normalize rotations to [-pi, pi] for stability
        delta_rot1 = (delta_rot1 + math.pi) % (2.0 * math.pi) - math.pi
        delta_rot2 = (delta_rot2 + math.pi) % (2.0 * math.pi) - math.pi

        self.log_particles("before odom motion update")
        self.mcl.motionUpdateOdometry(delta_rot1, delta_trans, delta_rot2)
        self.log_particles("after odom motion update")

        self.last_odom_pose = (x_curr, y_curr, theta_curr)


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

        # self.log_particles("before measurement update")
        self.mcl.measurementUpdate()
        # self.log_particles("after measurement udpate")

        # self.log_particles("before resampling")
        self.mcl.resampling()
        # self.log_particles("after_resampling")

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

        odom.pose.pose.orientation = MCLNode.yaw_to_quaternion(theta)

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

            q = MCLNode.yaw_to_quaternion(float(theta))
            pose.orientation = q

            msg.poses.append(pose)

        self.particles_pub.publish(msg)

    @staticmethod
    def yaw_to_quaternion(theta: float) -> Quaternion:
        half = 0.5 * float(theta)
        return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))
    
    @staticmethod        
    def quat_to_yaw(q: Quaternion) -> float:
        # yaw from quaternion (x,y,z,w)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def log_particles(self, stage: str, k_top=5, k_rand=5):
        P = self.mcl.Particles
        w = self.mcl.particle_weights

        if P is None or w is None:
            self.get_logger().warn(f"[{stage}] No particles or weights")
            return

        n = len(w)
        self.get_logger().info(f"\n[{stage}] N={n}")

        # sort by weight descending
        idx_sorted = np.argsort(w)[::-1]

        self.get_logger().info("Top particles by weight:")
        for i in range(min(k_top, n)):
            idx = idx_sorted[i]
            x, y, th = P[idx]
            self.get_logger().info(
                f"  {i+1:02d}) idx={idx:4d}  w={w[idx]:.6f}  p=({x:.3f},{y:.3f},{th:.3f})"
            )

        self.get_logger().info("Random particles:")
        rnd_idx = np.random.choice(n, size=min(k_rand, n), replace=False)
        for idx in rnd_idx:
            x, y, th = P[idx]
            self.get_logger().info(
                f"  idx={idx:4d}  w={w[idx]:.6f}  p=({x:.3f},{y:.3f},{th:.3f})"
            )




def main():
    rclpy.init()
    node = MCLNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
