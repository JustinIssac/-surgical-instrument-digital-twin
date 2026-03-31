import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import torch
from ultralytics import YOLO


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.declare_parameter('model_path',
            '/home/inoruske/surgical_twin_ws/models/best.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('device', 'cuda')
        model_path = self.get_parameter('model_path').value
        self.conf  = self.get_parameter('confidence_threshold').value
        device     = self.get_parameter('device').value
        self.get_logger().info(f'Loading model from {model_path}...')
        self.model = YOLO(model_path)
        self.model.to(device)
        self.get_logger().info('Model loaded successfully ✅')
        self.class_names = [
            'Large_Needle_Driver_Left',
            'Large_Needle_Driver_Right',
            'Prograsp_Forceps_Left',
            'Prograsp_Forceps_Right',
            'Maryland_Bipolar_Forceps',
            'Bipolar_Forceps',
            'Monopolar_Curved_Scissors',
            'Grasping_Retractor_Right'
        ]
        self.detection_pub = self.create_publisher(String, '/instrument_detections', 10)
        self.image_pub     = self.create_publisher(Image, '/annotated_frame', 10)
        self.image_sub     = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.bridge      = CvBridge()
        self.frame_count = 0
        self.get_logger().info('Perception node ready, waiting for frames...')

    def image_callback(self, msg):
        frame        = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img_h, img_w = frame.shape[:2]
        self.frame_count += 1
        results    = self.model(frame, conf=self.conf, verbose=False)[0]
        detections = []
        if results.masks is not None:
            for i, (box, mask) in enumerate(zip(results.boxes, results.masks)):
                class_id   = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox       = box.xyxy[0].tolist()
                mask_np      = mask.data[0].cpu().numpy()
                mask_resized = cv2.resize(mask_np, (img_w, img_h),
                                          interpolation=cv2.INTER_LINEAR)
                mask_bin = (mask_resized > 0.5).astype(np.uint8)
                moments  = cv2.moments(mask_bin)
                if moments['m00'] > 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                else:
                    cx = int((bbox[0] + bbox[2]) / 2)
                    cy = int((bbox[1] + bbox[3]) / 2)
                detections.append({
                    'class_id'   : class_id,
                    'class_name' : self.class_names[class_id],
                    'confidence' : round(confidence, 3),
                    'bbox'       : [round(x, 1) for x in bbox],
                    'centroid_px': [cx, cy],
                    'frame_id'   : self.frame_count
                })
        detection_msg      = String()
        detection_msg.data = json.dumps({
            'frame_id'       : self.frame_count,
            'timestamp'      : self.get_clock().now().to_msg().sec,
            'image_size'     : [img_w, img_h],
            'num_detections' : len(detections),
            'detections'     : detections
        })
        self.detection_pub.publish(detection_msg)
        annotated     = results.plot()
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        self.image_pub.publish(annotated_msg)
        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f'Frame {self.frame_count}: {len(detections)} instruments detected')


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down perception node...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
