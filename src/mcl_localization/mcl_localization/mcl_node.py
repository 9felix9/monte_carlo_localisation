import rclpy
from rclpy.node import Node

## some placeholder code for beginning and test of setup 

class MCLNode(Node):
    def __init__(self):
        super().__init__('mcl_node')
        self.get_logger().info("Monte Carlo Localization node started")


def main():
    rclpy.init()
    node = MCLNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
