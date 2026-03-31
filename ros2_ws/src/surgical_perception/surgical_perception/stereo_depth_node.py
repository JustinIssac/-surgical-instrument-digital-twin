import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import json


class StereoDepthNode(Node):
    def __init__(self):
        super().__init__('stereo_depth_node')

        # Camera intrinsics scaled to 1920x1080
        scale_x = 1920 / 1280
        scale_y = 1080 / 1024

        self.fx = 1068.39 * scale_x   # 1602.59
        self.fy = 1068.19 * scale_y   # 1126.94
        self.cx = 600.90  * scale_x   # 901.35
        self.cy = 500.74  * scale_y   # 528.28

        # Stereo baseline in metres (from calibration: -4.2773mm)
        self.baseline = abs(-4.2773) / 1000.0  # convert mm to metres

        # Distortion coefficients
        self.dist_coeffs = np.array(
            [-0.00087, 0.00238, 0.00012, 0.00000, 0.00000],
            dtype=np.float64
        )

        # Camera matrix
        self.K = np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1      ]
        ], dtype=np.float64)

        # Stereo matcher - Semi-Global Block Matching (SGBM)
        # Best balance of accuracy vs speed for surgical video
        self.stereo = cv2.StereoSGBM_create(
            minDisparity    = 0,
            numDisparities  = 64,    # max disparity range
            blockSize       = 7,     # matching block size
            P1              = 8  * 3 * 7 ** 2,
            P2              = 32 * 3 * 7 ** 2,
            disp12MaxDiff   = 1,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange    = 32,
            preFilterCap    = 63,
            mode            = cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        self.bridge      = CvBridge()
        self.left_frame  = None
        self.right_frame = None

        # Subscribers for left and right frames
        self.left_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.left_callback,
            10
        )
        self.right_sub = self.create_subscription(
            Image,
            '/camera/right/image_raw',
            self.right_callback,
            10
        )

        # Subscribe to detections to add stereo depth
        self.detection_sub = self.create_subscription(
            String,
            '/instrument_detections',
            self.detection_callback,
            10
        )

        # Publish depth-enhanced detections
        self.depth_pub = self.create_publisher(
            String,
            '/instrument_detections_3d',
            10
        )

        # Publish disparity image for visualisation
        self.disparity_pub = self.create_publisher(
            Image,
            '/disparity_image',
            10
        )

        self.get_logger().info(
            f'Stereo depth node ready ✅\n'
            f'  Baseline: {self.baseline*1000:.2f}mm\n'
            f'  fx={self.fx:.2f}, fy={self.fy:.2f}'
        )

    def left_callback(self, msg):
        self.left_frame = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='bgr8'
        )

    def right_callback(self, msg):
        self.right_frame = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='bgr8'
        )

    def compute_disparity(self):
        """Compute disparity map from stereo pair."""
        if self.left_frame is None or self.right_frame is None:
            return None

        # Convert to grayscale for stereo matching
        left_gray  = cv2.cvtColor(self.left_frame,  cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(self.right_frame, cv2.COLOR_BGR2GRAY)

        # Compute disparity
        disparity = self.stereo.compute(
            left_gray, right_gray
        ).astype(np.float32) / 16.0

        # Filter invalid disparities
        disparity[disparity <= 0] = np.nan

        return disparity

    def disparity_to_depth(self, disparity_value):
        """
        Convert disparity to depth using stereo formula:
        Z = (fx * baseline) / disparity
        """
        if disparity_value is None or np.isnan(disparity_value) or disparity_value <= 0:
            return None
        return (self.fx * self.baseline) / disparity_value

    def get_depth_at_centroid(self, disparity_map, cx, cy, window=5):
        """
        Sample disparity at centroid with a small window
        for robustness against noise.
        """
        if disparity_map is None:
            return None

        h, w = disparity_map.shape
        x1 = max(0, cx - window)
        x2 = min(w, cx + window)
        y1 = max(0, cy - window)
        y2 = min(h, cy + window)

        region = disparity_map[y1:y2, x1:x2]
        valid  = region[~np.isnan(region)]

        if len(valid) == 0:
            return None

        # Use median for robustness against outliers
        median_disparity = np.median(valid)
        return self.disparity_to_depth(median_disparity)

    def pixel_to_3d_stereo(self, cx_px, cy_px, depth_z):
        """
        Convert 2D pixel + stereo depth to 3D camera coordinates.
        """
        X = (cx_px - self.cx) * depth_z / self.fx
        Y = (cy_px - self.cy) * depth_z / self.fy
        return round(X, 4), round(Y, 4), round(depth_z, 4)

    def detection_callback(self, msg):
        data       = json.loads(msg.data)
        detections = data.get('detections', [])

        # Compute disparity map if stereo frames available
        disparity_map = self.compute_disparity()

        # Publish disparity visualisation
        if disparity_map is not None:
            disp_vis = cv2.normalize(
                np.nan_to_num(disparity_map),
                None, 0, 255,
                cv2.NORM_MINMAX, cv2.CV_8U
            )
            disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_PLASMA)
            disp_msg   = self.bridge.cv2_to_imgmsg(
                disp_color, encoding='bgr8'
            )
            self.disparity_pub.publish(disp_msg)

        enhanced_detections = []
        for det in detections:
            cx_px, cy_px = det['centroid_px']
            class_name   = det['class_name']

            # Try stereo depth first
            depth_z      = None
            depth_method = 'assumed'

            if disparity_map is not None:
                depth_z = self.get_depth_at_centroid(
                    disparity_map, cx_px, cy_px
                )
                if depth_z is not None:
                    # Sanity check: surgical instruments are 5-50cm away
                    if 0.05 <= depth_z <= 0.50:
                        depth_method = 'stereo'
                    else:
                        depth_z = None  # reject unrealistic depth

            # Fallback to assumed depth if stereo fails
            if depth_z is None:
                depth_z = 0.15
                depth_method = 'assumed'

            X, Y, Z = self.pixel_to_3d_stereo(cx_px, cy_px, depth_z)

            enhanced_detections.append({
                **det,
                'position_3d' : {'x': X, 'y': Y, 'z': Z},
                'depth_method': depth_method,
                'depth_m'     : round(depth_z, 4),
            })

        # Publish enhanced detections
        out_msg      = String()
        out_msg.data = json.dumps({
            'frame_id'  : data['frame_id'],
            'timestamp' : data['timestamp'],
            'image_size': data.get('image_size', [1920, 1080]),
            'detections': enhanced_detections,
            'stereo_available': disparity_map is not None
        })
        self.depth_pub.publish(out_msg)

        # Log stereo vs assumed usage
        stereo_count  = sum(
            1 for d in enhanced_detections
            if d['depth_method'] == 'stereo'
        )
        assumed_count = len(enhanced_detections) - stereo_count

        if enhanced_detections:
            self.get_logger().info(
                f'Frame {data["frame_id"]}: '
                f'{stereo_count} stereo depth, '
                f'{assumed_count} assumed depth'
            )


def main(args=None):
    rclpy.init(args=args)
    node = StereoDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stereo depth node shutting down...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
