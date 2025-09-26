import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
import cv2
import threading
import numpy as np
from typing import Optional

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')

        self.declare_parameter('qos_depth', 10)
        qos_depth = self.get_parameter('qos_depth').value

        QOS_RKL10V = QoSProfile(
            reliability = QoSReliabilityPolicy.RELIABLE,
            history = QoSHistoryPolicy.KEEP_LAST,
            depth=qos_depth,
            durability = QoSDurabilityPolicy.VOLATILE
        )
        self.publisher = self.create_publisher(Image, 'video_topic', QOS_RKL10V)
        self.bridge = CvBridge()

    def publish_video(self, frame):
            # OpenCV BGR 이미지를 ROS2 Image 메시지로 변환
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher.publish(msg)
        # 프레임 속도 조절 (예: 30fps)
        #rclpy.spin_once(self, timeout_sec=1/30)


class VideoSubscriber(Node):
    def __init__(self, topic_name: str = 'VehSensor_3/image_raw'):
        super().__init__('video_subscriber')
        
        self.bridge = CvBridge()
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        self.declare_parameter('qos_depth', 10)
        qos_depth = self.get_parameter('qos_depth').value
        
        QOS_RKL10V = QoSProfile(
            reliability = QoSReliabilityPolicy.RELIABLE,
            history = QoSHistoryPolicy.KEEP_LAST,
            depth=qos_depth,
            durability = QoSDurabilityPolicy.VOLATILE
        )
        
        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.image_callback,
            QOS_RKL10V
        )
        
        self.get_logger().info(f'Video subscriber started, listening to: {topic_name}')
    
    def image_callback(self, msg):
        try:
            # ROS2 Image 메시지를 OpenCV 프레임으로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            with self.frame_lock:
                self.current_frame = cv_image.copy()
                
        except Exception as e:
            self.get_logger().error(f"Error converting ROS2 image to OpenCV frame: {str(e)}")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def is_frame_available(self) -> bool:
        with self.frame_lock:
            return self.current_frame is not None
