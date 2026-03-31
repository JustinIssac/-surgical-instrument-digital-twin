import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import json


class PoseEstimatorNode(Node):
    def __init__(self):
        super().__init__('pose_estimator')

        # Camera intrinsics scaled from 1280x1024 to 1920x1080
        self.fx = 1068.39 * (1920 / 1280)   # 1602.59
        self.fy = 1068.19 * (1080 / 1024)   # 1126.94
        self.cx = 600.90  * (1920 / 1280)   # 901.35
        self.cy = 500.74  * (1080 / 1024)   # 528.28

        # Distortion coefficients [k1, k2, p1, p2, k3]
        self.dist_coeffs = np.array(
            [-0.00087, 0.00238, 0.00012, 0.00000, 0.00000],
            dtype=np.float64
        )

        # Assumed depth per instrument type in metres
        # Used as fallback when stereo depth is unavailable
        self.assumed_depth = {
            'Large_Needle_Driver_Left'  : 0.15,
            'Large_Needle_Driver_Right' : 0.15,
            'Prograsp_Forceps_Left'     : 0.15,
            'Prograsp_Forceps_Right'    : 0.15,
            'Maryland_Bipolar_Forceps'  : 0.12,
            'Bipolar_Forceps'           : 0.12,
            'Monopolar_Curved_Scissors' : 0.18,
            'Grasping_Retractor_Right'  : 0.15,
        }
        self.default_depth = 0.15

        # Subscribe to stereo-enhanced detections
        self.sub = self.create_subscription(
            String,
            '/instrument_detections_3d',
            self.detection_callback,
            10
        )

        # Publish 3D poses
        self.pub = self.create_publisher(
            String,
            '/instrument_poses_3d',
            10
        )

        self.get_logger().info(
            f'Pose estimator ready ✅\n'
            f'  fx={self.fx:.2f}, fy={self.fy:.2f}\n'
            f'  cx={self.cx:.2f}, cy={self.cy:.2f}\n'
            f'  Subscribed to: /instrument_detections_3d (stereo-enhanced)'
        )

    def pixel_to_3d(self, cx_px, cy_px, class_name):
        """
        Fallback: Convert 2D pixel centroid to 3D using assumed depth.
        Used when stereo depth is unavailable.
        Formula: X = (u - cx) * Z / fx
                 Y = (v - cy) * Z / fy
                 Z = assumed depth
        """
        Z      = self.assumed_depth.get(class_name, self.default_depth)
        u_norm = (cx_px - self.cx) / self.fx
        v_norm = (cy_px - self.cy) / self.fy
        X      = u_norm * Z
        Y      = v_norm * Z
        return round(X, 4), round(Y, 4), round(Z, 4)

    def estimate_orientation(self, bbox):
        """
        Estimate instrument orientation from bounding box aspect ratio.
        Returns roll, pitch, yaw in radians.
        Simplified estimate — Phase 5 replaces with full PnP.
        """
        x1, y1, x2, y2 = bbox
        width  = x2 - x1
        height = y2 - y1
        yaw    = 0.0 if width > height else 1.5708
        return 0.0, 0.0, round(yaw, 4)

    def detection_callback(self, msg):
        data       = json.loads(msg.data)
        detections = data.get('detections', [])
        frame_id   = data.get('frame_id', 0)

        poses = []
        for det in detections:
            class_name   = det['class_name']
            bbox         = det['bbox']
            depth_method = det.get('depth_method', 'assumed')

            # Use stereo 3D position if already computed by stereo_depth_node
            if 'position_3d' in det and depth_method == 'stereo':
                pos  = det['position_3d']
                X, Y, Z = pos['x'], pos['y'], pos['z']
            else:
                # Fallback: compute from centroid + assumed depth
                cx_px, cy_px = det['centroid_px']
                X, Y, Z      = self.pixel_to_3d(cx_px, cy_px, class_name)
                depth_method = 'assumed'

            roll, pitch, yaw = self.estimate_orientation(bbox)

            poses.append({
                'class_id'    : det['class_id'],
                'class_name'  : class_name,
                'confidence'  : det['confidence'],
                'position_3d' : {
                    'x': round(X, 4),
                    'y': round(Y, 4),
                    'z': round(Z, 4)
                },
                'orientation' : {
                    'roll' : roll,
                    'pitch': pitch,
                    'yaw'  : yaw
                },
                'centroid_px' : det['centroid_px'],
                'depth_method': depth_method,
                'frame_id'    : frame_id
            })

        # Publish 3D poses
        out_msg      = String()
        out_msg.data = json.dumps({
            'frame_id' : frame_id,
            'timestamp': data.get('timestamp', 0),
            'poses'    : poses
        })
        self.pub.publish(out_msg)

        if poses:
            stereo_count  = sum(
                1 for p in poses if p['depth_method'] == 'stereo'
            )
            assumed_count = len(poses) - stereo_count
            self.get_logger().info(
                f'Frame {frame_id}: {len(poses)} poses published '
                f'({stereo_count} stereo depth, {assumed_count} assumed depth)'
            )


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Pose estimator shutting down...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
