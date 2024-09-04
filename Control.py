#!/usr/bin/env python3

import math
from abc import ABC, abstractmethod
from copy import copy
import cv2
import numpy as np
import rclpy
from cv2.aruco import ArucoDetector
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Twist
from mavros.base import SENSOR_QOS, STATE_QOS
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image

class Controller(ABC):
    @property
    @abstractmethod
    def position(self) -> np.ndarray:
        pass
    @property
    @abstractmethod
    def orientation(self) -> np.ndarray:
        pass
    @property
    @abstractmethod
    def image(self) -> np.ndarray:
        pass
    @property
    @abstractmethod
    def image_viz(self) -> np.ndarray:
        pass
    @property
    @abstractmethod
    def goal_position(self) -> np.ndarray:
        pass
    @goal_position.setter
    @abstractmethod
    def goal_position(self, value: np.ndarray) -> None:
        pass
    @property
    @abstractmethod
    def goal_yaw(self) -> float:
        pass
    @goal_yaw.setter
    @abstractmethod
    def goal_yaw(self, value: float) -> None:
        pass
    @property
    def goal_distance(self) -> float:
        return np.linalg.norm(self.goal_position - self.position)
class Step(ABC):
    def init(self, controller: Controller) -> None:
        pass
    @abstractmethod
    def update(self, controller: Controller) -> bool:
        '''Returns: True if the step was completed, False otherwise.'''
        pass
    def __str__(self) -> str:
        return self.__class__.__name__
class BasicController(Node):
    def __init__(self, node: Node, task_list: list[Step], frequency: float = 100.0):
        super().__init__('basic_controller')
        self.node = node
        self.task_list = task_list
        self.current_step: Step | None = None
        self.current_pose: Pose | None = None
        self.target_pose: Pose | None = None
        self.image_data: np.ndarray | None = None
        self.visual_image: np.ndarray | None = None
        
        mavros_namespace = '/mavros'
        self.cv_bridge = CvBridge()
        camera_topic = '/camera'
        camera_depth = 10
        self.camera_subscriber = node.create_subscription(Image, camera_topic, self.camera_data_callback, camera_depth)
        
        period = 1.0 / frequency
        self.timer = self.node.create_timer(period, self.timer_callback)
        self.velocity_publisher = node.create_publisher(Twist, mavros_namespace + '/setpoint_velocity/cmd_vel_unstamped', qos_profile=SENSOR_QOS)
        self.pose_subscriber = node.create_subscription(PoseStamped, mavros_namespace + '/local_position/pose', self.pose_data_callback, qos_profile=SENSOR_QOS)
        self.arming_service = self.initialize_service(CommandBool, mavros_namespace + '/cmd/arming')
        self.state_subscriber = node.create_subscription(State, mavros_namespace + '/state', self.state_data_callback, qos_profile=STATE_QOS)
        self.goal_pose_publisher = node.create_publisher(PoseStamped, mavros_namespace + '/setpoint_position/local', qos_profile=SENSOR_QOS)
        self.mode_service = self.initialize_service(SetMode, mavros_namespace + '/set_mode')

    def camera_data_callback(self, msg: Image):
        try:
            self.image_data = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

    def pose_data_callback(self, msg: PoseStamped):
        self.current_pose = msg.pose
        if self.target_pose is None:
            self.target_pose = copy(self.current_pose)

    def state_data_callback(self, msg):
        if msg.mode != State.MODE_PX4_OFFBOARD:
            request = SetMode.Request(custom_mode='OFFBOARD')
            self.mode_service.call_async(request)
        elif not msg.armed:
            request = CommandBool.Request(value=True)
            self.arming_service.call_async(request)

    def initialize_service(self, srv_type, srv_name):
        client = self.create_client(srv_type, srv_name)
        while not client.wait_for_service(timeout_sec=1.0):
            pass  
        return client

def timer_callback(self):
    if self.current_pose is None or self.image_data is None:
        return

    self.visual_image = self.image_data.copy()
    
    if self.current_step is None:
        if not self.task_list:
            raise SystemExit
        self.current_step = self.task_list.pop(0)
        self.current_step.initialize(self)

    step_description = str(self.current_step)
    if self.current_step.update(self):
        self.current_step = None

    self.goal_pose_publisher.publish(PoseStamped(pose=self.target_pose))

    def format_vector(vector: np.ndarray) -> str:
        return np.array2string(vector, formatter={'float_kind': lambda x: "%.2f" % x})[1:-1]

    status_lines = [
        f"Step info: {step_description}",
        f"Current Pos: {format_vector(self.position)}",
        f"Gate Pos: {format_vector(self.goal_position)}"
    ]

    for index, line in enumerate(status_lines):
        # Place text in the bottom-left corner
        cv2.putText(self.visual_image, line, (10, self.visual_image.shape[0] - 10 - 32 * (len(status_lines) - index)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)

    cv2.imshow("Basic Controller", self.visual_image)
    cv2.waitKey(1)

    @property
    def position(self) -> np.ndarray:
        p = self._pose.position
        return np.array([p.x, p.y, p.z])
    @property
    def orientation(self) -> np.ndarray:
        o = self._pose.orientation
        return np.array([o.x, o.y, o.z, o.w])
    @property
    def image(self) -> np.ndarray:
        return self._image
    @property
    def image_viz(self) -> np.ndarray:
        return self._image_viz
    @property
    def goal_position(self) -> np.ndarray:
        p = self._goal_pose.position
        return np.array([p.x, p.y, p.z])
    @goal_position.setter
    def goal_position(self, value: np.ndarray) -> None:
        self._goal_pose.position = Point(x=value[0], y=value[1], z=value[2])
    @property
    def goal_yaw(self) -> float:
        o = self._goal_pose.orientation
        quat = np.array([o.x, o.y, o.z, o.w])
        return Rotation.from_quat(quat).as_euler('xyz')[2]
    @goal_yaw.setter
    def goal_yaw(self, value: float):
        quat = Rotation.from_euler('xyz', np.array([0, 0, value])).as_quat()
        self._goal_pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

class Ascend(Step):
    def __init__(self, target_altitude: float, threshold: float = 0.2) -> None:
        self.target_altitude = target_altitude
        self.threshold = threshold

    def initialize(self, controller: Controller) -> None:
        controller.desired_position = controller.current_position + np.array([0, 0, self.target_altitude])

    def process(self, controller: Controller) -> bool:
        return controller.current_distance_to_goal < self.threshold


class Navigate(Step):
    def __init__(self, target_x: float, target_y: float, target_z: float, target_yaw: float | None = None, threshold: float = 0.5) -> None:
        self.target_position = np.array([target_x, target_y, target_z])
        self.target_yaw = target_yaw
        self.threshold = threshold

    def initialize(self, controller: Controller) -> None:
        controller.desired_position = self.target_position
        if self.target_yaw is not None:
            controller.desired_yaw = self.target_yaw

    def process(self, controller: Controller) -> bool:
        return controller.current_distance_to_goal < self.threshold

class GatePassageController(Step):
    def __init__(self, gate_index: int, entry_distance: float = 1.5, entry_margin: float = 0.5, exit_margin: float = 0.5) -> None:
        self.gate_index = gate_index
        self.entry_margin = entry_margin
        self.exit_margin = exit_margin
        self.passing_through_gate = False
        self.entry_distance = entry_distance

    def initialize(self, controller: Controller) -> None:
        aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_detector = ArucoDetector(aruco_dictionary)
        
        marker_size = 0.19
        gate_size = 1.54
        
        marker_offset = marker_size / 2
        gate_offset = (marker_size + gate_size) / 2
        
        self.marker_positions = np.array([
            [-marker_offset, -marker_offset, 0],
            [ marker_offset, -marker_offset, 0],
            [ marker_offset,  marker_offset, 0],
            [-marker_offset,  marker_offset, 0],
        ])
        self.gate_positions = np.array([
            [-gate_offset,  gate_offset, 0],
            [ gate_offset,  gate_offset, 0],
            [ gate_offset, -gate_offset, 0],
            [-gate_offset, -gate_offset, 0]
        ])
        
        img_width = 1280
        img_height = 720
        vertical_fov = math.radians(86.8)
        focal_length = img_height / (2 * math.tan(vertical_fov / 2))
        center_x = img_width / 2
        center_y = img_height / 2
        self.camera_intrinsics = np.array([
            [focal_length, 0, center_x],
            [0, focal_length, center_y],
            [0, 0, 1]
        ])
        
        self.entry_offset_optical = np.array([0, 0, -self.entry_distance])
        self.exit_offset_optical = np.array([0, 0.3, self.entry_distance])
        self.optical_to_camera_transform = np.array([
            [ 0,  0,  1,  0],
            [-1,  0,  0,  0],
            [ 0, -1,  0,  0],
            [ 0,  0,  0,  1],
        ])
        
        self.camera_to_base_transform = np.eye(4)
        self.camera_to_base_transform[:3, :3] = Rotation.from_euler('y', -20, degrees=True).as_matrix()
        self.camera_to_base_transform[0, 3] = 0.107

    def update_status(self, controller: Controller) -> bool:
        if self.passing_through_gate:
            return controller.goal_distance < self.exit_margin
        
        detected_corners, detected_ids, _ = self.aruco_detector.detectMarkers(controller.image)
        cv2.aruco.drawDetectedMarkers(controller.image_viz, detected_corners, detected_ids)
        
        image_points = np.empty((0, 2))
        object_points = np.empty((0, 3))
        
        if detected_ids is not None:
            for marker_id, corners in zip(np.squeeze(detected_ids, axis=1), np.squeeze(detected_corners, axis=1)):
                if marker_id // 4 == self.gate_index:
                    image_points = np.concatenate((image_points, corners))
                    object_points = np.concatenate((object_points, self.gate_positions[marker_id % 4] + self.marker_positions))
        
        if image_points.size == 0:
            return False
        
        dist_coeffs = None
        success, gate_rotation_vector, gate_translation_vector = cv2.solvePnP(object_points, image_points, self.camera_intrinsics, dist_coeffs)
        
        if not success:
            return False
        
        gate_rotation_matrix, _ = cv2.Rodrigues(gate_rotation_vector)
        gate_position_optical = np.squeeze(gate_translation_vector)
        
        entry_position_optical = gate_position_optical + gate_rotation_matrix @ self.entry_offset_optical
        exit_position_optical = gate_position_optical + gate_rotation_matrix @ self.exit_offset_optical
        
        for tvec in [gate_position_optical, entry_position_optical, exit_position_optical]:
            if tvec[2] > 0 and np.linalg.norm(tvec) >= 0.1:
                cv2.drawFrameAxes(controller.image_viz, self.camera_intrinsics, dist_coeffs, gate_rotation_vector, tvec, 0.2)
        
        world_transform = np.eye(4)
        world_transform[:3, :3] = Rotation.from_quat(controller.orientation).as_matrix()
        world_transform[:3, 3] = controller.position
        
        optical_to_world_transform = world_transform @ self.camera_to_base_transform @ self.optical_to_camera_transform
        
        gate_position_world = optical_to_world_transform @ np.append(gate_position_optical, 1)
        entry_position_world = optical_to_world_transform @ np.append(entry_position_optical, 1)
        exit_position_world = optical_to_world_transform @ np.append(exit_position_optical, 1)
        
        controller.goal_position = entry_position_world
        controller.goal_yaw = math.atan2(gate_position_world[1] - controller.position[1], gate_position_world[0] - controller.position[0])
        
        if controller.goal_distance < self.entry_margin:
            controller.goal_position = exit_position_world
            self.passing_through_gate = True
        
        return False



    def __str__(self) -> str:
        return '%s: %d (%s)' % (super().__str__(), self.gate_id, 'passing' if self.is_passing else 'approaching')
    
def main(args=None):
    rclpy.init(args=args)
    node = Node('BasicControlleer')
    
    # Определяем параметры для каждого круга
    circles = [
        {
            'points': [
                (248.3, 244.8, 196.3, 1.22, 0.7),
                (243.3, 254.0, 194.9, 1.57),
                (249.0, 258.8, 194.7, 5.60),
                (254.1, 248.5, 196.9, 4.73),
                (254.1, 241.1, 195.3, 1.57),
                (253.8, 252.2, 194.9, 2.80),
                (240.0, 249.8, 194.9, 5.00),
            ],
            'gate_indexes': [0, 1, 2, 3, 4, 5, 6]
        },
        {
            'points': [
                (240.0, 240.0, 200.0, 2.50, 0.8),
                (245.0, 245.0, 195.0, 1.80),
                (250.0, 250.0, 190.0, 4.50),
                (255.0, 255.0, 185.0, 3.60),
                (260.0, 260.0, 180.0, 2.00),
                (265.0, 265.0, 175.0, 1.50),
                (270.0, 270.0, 170.0, 1.00),
            ],
            'gate_indexes': [0, 1, 2, 3, 4, 5, 6]
        },
        {
            'points': [
                (230.0, 230.0, 210.0, 1.90, 0.7),
                (235.0, 235.0, 205.0, 1.70),
                (240.0, 240.0, 200.0, 4.30),
                (245.0, 245.0, 195.0, 3.50),
                (250.0, 250.0, 190.0, 2.30),
                (255.0, 255.0, 185.0, 1.80),
                (260.0, 260.0, 180.0, 1.20),
            ],
            'gate_indexes': [0, 1, 2, 3, 4, 5, 6]
        }
    ]

    # Создаем список шагов
    steps = []

    # Добавляем шаги для каждого круга
    for circle in circles:
        for index, (x, y, z, yaw, tolerance) in enumerate(circle['points']):
            steps.append(Navigate(x, y, z, yaw, tolerance))
            steps.append(GatePassageController(circle['gate_indexes'][index], index + 1, 0.8))

    # Создание контроллера с шагами
    controller = BasicController(node, steps)
    
    # Запуск основного цикла
    rclpy.spin(node)

    # Завершение работы
    rclpy.shutdown()

    simple_controller = BasicController(node, steps)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()
