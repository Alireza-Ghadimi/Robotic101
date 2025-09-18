import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def wrap_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class MoveToGoal(Node):
    def __init__(self):
        super().__init__('move_robot_to_goal')
        self.get_logger().info(f'{self.get_name()} created')

        # --- Parameters (position + heading) ---
        self.declare_parameter('goal_x', 0.0)
        self._goal_x = self.get_parameter('goal_x').get_parameter_value().double_value

        self.declare_parameter('goal_y', 0.0)
        self._goal_y = self.get_parameter('goal_y').get_parameter_value().double_value

        self.declare_parameter('goal_t', 0.0)  # desired yaw (radians)
        self._goal_t = self.get_parameter('goal_t').get_parameter_value().double_value

        # Optional tunables as parameters (nice for runtime tweaks)
        self.declare_parameter('vel_gain', 5.0)     # linear gain
        self.declare_parameter('max_vel', 0.2)      # linear saturation [m/s]
        self.declare_parameter('max_pos_err', 0.05) # [m]
        self.declare_parameter('ang_gain', 2.0)     # angular gain
        self.declare_parameter('max_omega', 1.0)    # angular saturation [rad/s]
        self.declare_parameter('max_theta_err', 0.05)  # [rad]

        self.add_on_set_parameters_callback(self.parameter_callback)
        self.get_logger().info(f"initial goal {self._goal_x} {self._goal_y} {self._goal_t}")

        self._subscriber = self.create_subscription(Odometry, "/odom", self._listener_callback, 1)
        self._publisher = self.create_publisher(Twist, "/cmd_vel", 1)

    def _listener_callback(self, msg):
        # Fetch dynamic params each callback so runtime changes take effect
        vel_gain     = self.get_parameter('vel_gain').value
        max_vel      = self.get_parameter('max_vel').value
        max_pos_err  = self.get_parameter('max_pos_err').value
        ang_gain     = self.get_parameter('ang_gain').value
        max_omega    = self.get_parameter('max_omega').value
        max_theta_err= self.get_parameter('max_theta_err').value

        pose = msg.pose.pose
        cur_x = pose.position.x
        cur_y = pose.position.y
        roll, pitch, yaw = euler_from_quaternion(pose.orientation)
        cur_t = yaw

        x_diff = self._goal_x - cur_x
        y_diff = self._goal_y - cur_y
        dist   = math.hypot(x_diff, y_diff)

        twist = Twist()

        if dist > max_pos_err:
            # --- POSITION CONTROL (holonomic) ---
            x_cmd = max(min(x_diff * vel_gain,  max_vel), -max_vel)
            y_cmd = max(min(y_diff * vel_gain,  max_vel), -max_vel)

            # World -> body transform using current yaw
            twist.linear.x =  x_cmd * math.cos(cur_t) + y_cmd * math.sin(cur_t)
            twist.linear.y = -x_cmd * math.sin(cur_t) + y_cmd * math.cos(cur_t)

            # Keep rotation calm while far from goal position
            twist.angular.z = 0.0

            self.get_logger().debug(
                f"pos ctrl at ({cur_x:.2f},{cur_y:.2f},{cur_t:.2f}) -> "
                f"goal ({self._goal_x:.2f},{self._goal_y:.2f},{self._goal_t:.2f})"
            )
        else:
            # --- ORIENTATION CONTROL (use goal_t) ---
            twist.linear.x = 0.0
            twist.linear.y = 0.0

            theta_err = wrap_pi(self._goal_t - cur_t)
            if abs(theta_err) > max_theta_err:
                w = max(min(ang_gain * theta_err, max_omega), -max_omega)
                twist.angular.z = w
                self.get_logger().debug(
                    f"orient ctrl yaw={cur_t:.2f} -> goal_t={self._goal_t:.2f} err={theta_err:.2f} w={w:.2f}"
                )
            else:
                twist.angular.z = 0.0
                self.get_logger().info("Goal reached: position and heading.")

        self._publisher.publish(twist)

    def parameter_callback(self, params):
        # Allow live updates to goal_* and gains
        for param in params:
            if param.name == 'goal_x' and param.type_ == Parameter.Type.DOUBLE:
                self._goal_x = param.value
            elif param.name == 'goal_y' and param.type_ == Parameter.Type.DOUBLE:
                self._goal_y = param.value
            elif param.name == 'goal_t' and param.type_ == Parameter.Type.DOUBLE:
                self._goal_t = param.value
            else:
                # Let other params (gains) be handled by dynamic get_parameter in callback
                # If you want to enforce types here, you can extend this section.
                continue
        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = MoveToGoal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()
