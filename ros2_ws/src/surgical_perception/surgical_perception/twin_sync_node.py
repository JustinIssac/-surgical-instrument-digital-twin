import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import gz.transport13
import gz.msgs10.entity_factory_pb2 as entity_factory_pb2
import gz.msgs10.boolean_pb2 as boolean_pb2
import gz.msgs10.pose_pb2 as pose_pb2
import json
import math
import numpy as np


class InstrumentKalmanFilter:
    """
    3D Kalman filter for a single surgical instrument.
    State vector: [x, y, z, vx, vy, vz]
    - Position (x, y, z) in metres
    - Velocity (vx, vy, vz) in metres/second
    """

    def __init__(self, initial_pos, dt=0.1):
        self.dt = dt  # time step (10 FPS = 0.1s)

        # State vector [x, y, z, vx, vy, vz]
        self.x = np.array([
            initial_pos[0], initial_pos[1], initial_pos[2],
            0.0, 0.0, 0.0
        ], dtype=np.float64)

        # State transition matrix F
        # Models: new_pos = old_pos + velocity * dt
        self.F = np.array([
            [1, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0,  dt, 0 ],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0,  0,  1 ],
        ], dtype=np.float64)

        # Observation matrix H
        # We only observe position, not velocity
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=np.float64)

        # Process noise Q - how much we trust the motion model
        # Higher = allows faster movements
        q = 0.01
        self.Q = np.eye(6, dtype=np.float64) * q

        # Measurement noise R - how much we trust the detector
        # Higher = smoother but slower to respond
        r = 0.005
        self.R = np.eye(3, dtype=np.float64) * r

        # Initial covariance P
        self.P = np.eye(6, dtype=np.float64) * 0.1

        # Track consecutive missed detections
        self.missed_frames   = 0
        self.max_missed      = 10  # remove after 10 missed frames
        self.is_active       = True

    def predict(self):
        """Predict next state based on motion model."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3]  # return predicted position

    def update(self, measurement):
        """
        Update state with new measurement.
        measurement: [x, y, z] observed position
        """
        z = np.array(measurement, dtype=np.float64)

        # Innovation (difference between measurement and prediction)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain - how much to trust measurement vs prediction
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

        self.missed_frames = 0
        return self.x[:3]  # return filtered position

    def mark_missed(self):
        """Call when no detection found for this instrument."""
        self.missed_frames += 1
        if self.missed_frames > self.max_missed:
            self.is_active = False
        return self.predict()  # keep predicting even when missed

    @property
    def position(self):
        return self.x[:3]

    @property
    def velocity(self):
        return self.x[3:]


class TwinSyncNode(Node):
    def __init__(self):
        super().__init__('twin_sync_node')

        self.gz_node = gz.transport13.Node()

        self.instrument_models = {
            0: 'instrument_needle_left',
            1: 'instrument_needle_right',
            2: 'instrument_prograsp_left',
            3: 'instrument_prograsp_right',
            4: 'instrument_maryland',
            5: 'instrument_bipolar',
            6: 'instrument_scissors',
            7: 'instrument_retractor',
        }

        self.spawned_models = set()

        # Kalman filters — one per instrument class
        self.kalman_filters = {}

        # Track path length for Economy of Motion (Phase 5)
        self.path_lengths = {}
        self.last_positions = {}

        self.class_colors = {
            0: (0.2, 0.6, 1.0),
            1: (1.0, 0.4, 0.2),
            2: (0.2, 1.0, 0.4),
            3: (1.0, 0.2, 0.8),
            4: (0.8, 0.8, 0.2),
            5: (0.6, 0.2, 1.0),
            6: (0.2, 0.8, 0.8),
            7: (1.0, 0.6, 0.2),
        }

        self.sdf_template = """<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{name}">
    <static>true</static>
    <link name="shaft">
      <visual name="shaft_visual">
        <geometry>
          <cylinder><radius>0.005</radius><length>0.35</length></cylinder>
        </geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
        </material>
      </visual>
    </link>
    <link name="tip">
      <pose>0 0 0.18 0 0 0</pose>
      <visual name="tip_visual">
        <geometry><sphere><radius>0.008</radius></sphere></geometry>
        <material>
          <ambient>1.0 1.0 0.0 1</ambient>
          <diffuse>1.0 1.0 0.0 1</diffuse>
        </material>
      </visual>
    </link>
    <joint name="shaft_tip_joint" type="fixed">
      <parent>shaft</parent><child>tip</child>
    </joint>
  </model>
</sdf>"""

        # Publisher for filtered poses (useful for evaluation)
        self.filtered_pub = self.create_publisher(
            String,
            '/instrument_poses_filtered',
            10
        )

        self.sub = self.create_subscription(
            String,
            '/instrument_poses_3d',
            self.pose_callback,
            10
        )

        self.frame_count = 0
        self.get_logger().info('Twin sync node with Kalman filter ready ✅')

    def spawn_instrument(self, class_id, model_name):
        r, g, b = self.class_colors.get(class_id, (0.7, 0.7, 0.7))
        sdf     = self.sdf_template.format(name=model_name, r=r, g=g, b=b)

        req      = entity_factory_pb2.EntityFactory()
        req.sdf  = sdf
        req.name = model_name
        req.pose.position.x = 0.0
        req.pose.position.y = 0.0
        req.pose.position.z = 0.5

        result = self.gz_node.request(
            '/world/empty/create',
            req,
            entity_factory_pb2.EntityFactory,
            boolean_pb2.Boolean,
            2000
        )

        if result:
            self.spawned_models.add(model_name)
            self.get_logger().info(f'Spawned: {model_name}')
        else:
            self.get_logger().warn(f'Failed to spawn: {model_name}')

    def move_instrument(self, model_name, x, y, z, roll, pitch, yaw):
        pose_msg            = pose_pb2.Pose()
        pose_msg.name       = model_name
        pose_msg.position.x = float(x)
        pose_msg.position.y = float(y)
        pose_msg.position.z = float(z)

        cy = math.cos(yaw   * 0.5)
        sy = math.sin(yaw   * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll  * 0.5)
        sr = math.sin(roll  * 0.5)

        pose_msg.orientation.w = cr * cp * cy + sr * sp * sy
        pose_msg.orientation.x = sr * cp * cy - cr * sp * sy
        pose_msg.orientation.y = cr * sp * cy + sr * cp * sy
        pose_msg.orientation.z = cr * cp * sy - sr * sp * cy

        self.gz_node.request(
            '/world/empty/set_pose',
            pose_msg,
            pose_pb2.Pose,
            boolean_pb2.Boolean,
            500
        )

    def update_path_length(self, class_id, new_pos):
        """Track cumulative path length for Economy of Motion."""
        if class_id in self.last_positions:
            last = self.last_positions[class_id]
            dist = np.linalg.norm(
                np.array(new_pos) - np.array(last)
            )
            self.path_lengths[class_id] = (
                self.path_lengths.get(class_id, 0.0) + dist
            )
        self.last_positions[class_id] = new_pos

    def pose_callback(self, msg):
        data       = json.loads(msg.data)
        poses      = data.get('poses', [])
        frame_id   = data.get('frame_id', 0)
        self.frame_count += 1

        # Track which class IDs were detected this frame
        detected_ids  = set()
        filtered_poses = []

        for pose in poses:
            class_id   = pose['class_id']
            class_name = pose['class_name']
            pos        = pose['position_3d']
            ori        = pose['orientation']
            conf       = pose['confidence']

            if conf < 0.6:
                continue

            detected_ids.add(class_id)
            model_name  = self.instrument_models.get(
                class_id, f'instrument_{class_id}'
            )
            measurement = [pos['x'], pos['y'], pos['z']]

            # Initialise Kalman filter on first detection
            if class_id not in self.kalman_filters:
                self.kalman_filters[class_id] = InstrumentKalmanFilter(
                    measurement
                )
                self.get_logger().info(
                    f'Initialised Kalman filter for {class_name}'
                )

            # Predict then update
            kf = self.kalman_filters[class_id]
            kf.predict()
            filtered_pos = kf.update(measurement)

            # Update path length
            self.update_path_length(class_id, filtered_pos.tolist())

            # Spawn if needed
            if model_name not in self.spawned_models:
                self.spawn_instrument(class_id, model_name)

            # Move to FILTERED position
            self.move_instrument(
                model_name,
                filtered_pos[0], filtered_pos[1], filtered_pos[2],
                ori['roll'], ori['pitch'], ori['yaw']
            )

            filtered_poses.append({
                'class_id'       : class_id,
                'class_name'     : class_name,
                'raw_position'   : [pos['x'], pos['y'], pos['z']],
                'filtered_position': filtered_pos.tolist(),
                'velocity'       : kf.velocity.tolist(),
                'path_length_m'  : round(
                    self.path_lengths.get(class_id, 0.0), 4
                ),
                'confidence'     : conf,
            })

            self.get_logger().info(
                f'[KF] {class_name}: '
                f'raw=({pos["x"]:.3f},{pos["y"]:.3f}) '
                f'filtered=({filtered_pos[0]:.3f},{filtered_pos[1]:.3f})'
            )

        # Run predict-only for instruments not detected this frame
        for class_id, kf in self.kalman_filters.items():
            if class_id not in detected_ids and kf.is_active:
                predicted = kf.mark_missed()
                model_name = self.instrument_models.get(class_id)
                if model_name and model_name in self.spawned_models:
                    self.move_instrument(
                        model_name,
                        predicted[0], predicted[1], predicted[2],
                        0.0, 0.0, 0.0
                    )

        # Publish filtered poses for evaluation
        out_msg      = String()
        out_msg.data = json.dumps({
            'frame_id'     : frame_id,
            'poses'        : filtered_poses,
            'path_lengths' : {
                str(k): round(v, 4)
                for k, v in self.path_lengths.items()
            }
        })
        self.filtered_pub.publish(out_msg)

        # Log path lengths every 100 frames
        if self.frame_count % 100 == 0 and self.path_lengths:
            self.get_logger().info('--- Economy of Motion ---')
            for cid, length in self.path_lengths.items():
                name = self.instrument_models.get(cid, f'class_{cid}')
                self.get_logger().info(
                    f'  {name}: {length*1000:.1f} mm total path'
                )


def main(args=None):
    rclpy.init(args=args)
    node = TwinSyncNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Twin sync node shutting down...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
