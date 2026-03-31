import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class Verifier(Node):
    def __init__(self):
        super().__init__('verifier')
        self.sub = self.create_subscription(
            String,
            '/instrument_detections',
            self.callback,
            10
        )
        self.count = 0

    def callback(self, msg):
        if self.count >= 3:  # print 3 messages then stop
            raise KeyboardInterrupt
        data = json.loads(msg.data)
        print(json.dumps(data, indent=2))
        print("---")
        self.count += 1

def main():
    rclpy.init()
    node = Verifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
