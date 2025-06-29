import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import numpy as np
from scipy.spatial.transform import Rotation as R
import heapq
from enum import Enum

from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, PoseArray, TransformStamped, Quaternion, PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from transforms3d.euler import quat2euler, euler2quat
import math

class State(Enum):
    IDLE = 0
    PLANNING = 1
    INITIAL_POSITIONING = 2
    NAVIGATING = 3
    AVOIDING_OBSTACLE = 4

class AmclNode(Node):
    def __init__(self):
        super().__init__('my_py_amcl')

        # --- Parameters ---
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_footprint')
        self.declare_parameter('map_frame_id', 'map')
        self.declare_parameter('scan_topic', 'scan')
        self.declare_parameter('map_topic', 'map')
        self.declare_parameter('initial_pose_topic', 'initialpose')
        self.declare_parameter('laser_max_range', 3.5)
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('obstacle_detection_distance', 0.22) #era 0.3
        self.declare_parameter('obstacle_avoidance_turn_speed', 0.2)

        # --- Parameters to set ---
        # OK: Setear valores default
        self.declare_parameter('num_particles', 100)
        self.declare_parameter('alpha1', 0.02)
        self.declare_parameter('alpha2', 0.02)
        self.declare_parameter('alpha3', 0.005)
        self.declare_parameter('alpha4', 0.005)
        self.declare_parameter('z_hit', 0.8)
        self.declare_parameter('z_rand', 0.2)
        self.declare_parameter('lookahead_distance', 0.3)
        self.declare_parameter('linear_velocity', 0.1)
        self.declare_parameter('goal_tolerance', 0.1)
        self.declare_parameter('path_pruning_distance', 0.1)
        self.declare_parameter('safety_margin_cells', 4) # era 2
        
        # --- Pure Pursuit Parameters ---
        self.declare_parameter('max_curvature', 2.0)
        self.declare_parameter('pure_pursuit_gain', 1.0)
        self.declare_parameter('min_lookahead_distance', 0.1)
        self.declare_parameter('max_lookahead_distance', 0.5)
        
        # --- Initial Positioning Parameters ---
        self.declare_parameter('initial_positioning_angular_speed', 0.3)
        self.declare_parameter('initial_positioning_tolerance', 0.075)
        
        self.num_particles = self.get_parameter('num_particles').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.map_frame_id = self.get_parameter('map_frame_id').value
        self.laser_max_range = self.get_parameter('laser_max_range').value
        self.z_hit = self.get_parameter('z_hit').value
        self.z_rand = self.get_parameter('z_rand').value
        self.alphas = np.array([
            self.get_parameter('alpha1').value,
            self.get_parameter('alpha2').value,
            self.get_parameter('alpha3').value,
            self.get_parameter('alpha4').value,
        ])
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.linear_velocity = self.get_parameter('linear_velocity').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.path_pruning_distance = self.get_parameter('path_pruning_distance').value
        self.safety_margin_cells = self.get_parameter('safety_margin_cells').value
        self.obstacle_detection_distance = self.get_parameter('obstacle_detection_distance').value
        self.obstacle_avoidance_turn_speed = self.get_parameter('obstacle_avoidance_turn_speed').value
        self.object_avoidance_direction = 1
        
        # --- Pure Pursuit Parameters ---
        self.max_curvature = self.get_parameter('max_curvature').value
        self.pure_pursuit_gain = self.get_parameter('pure_pursuit_gain').value
        self.min_lookahead_distance = self.get_parameter('min_lookahead_distance').value
        self.max_lookahead_distance = self.get_parameter('max_lookahead_distance').value
        
        # --- Initial Positioning Parameters ---
        self.initial_positioning_angular_speed = self.get_parameter('initial_positioning_angular_speed').value
        self.initial_positioning_tolerance = self.get_parameter('initial_positioning_tolerance').value
        
        # --- State ---
        self.particles = np.zeros((self.num_particles, 3))
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.map_data = None
        self.latest_scan = None
        self.initial_pose_received = False
        self.map_received = False
        self.last_odom_pose = None
        self.state = State.IDLE
        self.current_path = None
        self.goal_pose = None
        self.inflated_grid = None
        self.obstacle_avoidance_start_yaw = None
        self.obstacle_avoidance_last_yaw = None
        self.obstacle_avoidance_cumulative_angle = 0.0
        self.obstacle_avoidance_active = False
        self.distance_map = None
        
        # Variables for initial positioning
        self.target_orientation = None
        
        # Variables para suavizar el control
        self.last_angular_cmd = 0.0
        self.angular_filter_alpha = 0.5  # Factor de suavizado (0-1, menor = más suave)
        
        # --- ROS 2 Interfaces ---
        map_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        scan_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        
        self.map_sub = self.create_subscription(OccupancyGrid, self.get_parameter('map_topic').value, self.map_callback, map_qos)
        self.scan_sub = self.create_subscription(LaserScan, self.get_parameter('scan_topic').value, self.scan_callback, scan_qos)
        self.initial_pose_sub = self.create_subscription(PoseWithCovarianceStamped, self.get_parameter('initial_pose_topic').value, self.initial_pose_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, self.get_parameter('goal_topic').value, self.goal_callback, 10)
        
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'amcl_pose', 10)
        self.particle_pub = self.create_publisher(MarkerArray, 'particle_cloud', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, self.get_parameter('cmd_vel_topic').value, 10)
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('MyPyAMCL node initialized.')

    def map_callback(self, msg):
        if not self.map_received:
            self.map_data = msg
            self.map_received = True
            self.grid = np.array(self.map_data.data).reshape((self.map_data.info.height, self.map_data.info.width))
            self.inflate_map()
            self.precompute_distance_map()
            
            self.get_logger().info('Map, inflated map')

    def scan_callback(self, msg):
        self.latest_scan = msg

    def goal_callback(self, msg):
        if self.map_data is None:
            self.get_logger().warn("Goal received, but map is not available yet. Ignoring goal.")
            return

        if msg.header.frame_id != self.map_frame_id:
            self.get_logger().warn(f"Goal received in frame '{msg.header.frame_id}', but expected '{self.map_frame_id}'. Ignoring.")
            return
            
        self.goal_pose = msg.pose
        self.get_logger().info(f"New goal received: ({self.goal_pose.position.x:.2f}, {self.goal_pose.position.y:.2f}). State -> PLANNING")
        self.state = State.PLANNING
        self.current_path = None
        self.target_orientation = None  # Reset target orientation for new goal
        
    def plan_path(self, start, goal):
        """  A* path planning algorithm to find a path from start to goal in the occupancy grid. """
        width = self.map_data.info.width
        height = self.map_data.info.height
        grid = self.inflated_grid

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1]) # heruistica --> Manhattan distance

        def get_neighbors(node):
            neighbors = []
            # for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = node[0] + dx, node[1] + dy
                if 0 <= nx < width and 0 <= ny < height and grid[ny, nx] < 50: # VER ESTO DE 50 
                    neighbors.append((nx, ny))
            return neighbors

        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break

            for neighbor in get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(goal, neighbor)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        if goal not in came_from:
            return None

        path = Path()
        current = goal
        while current:
            wx, wy = self.grid_to_world(*current)
            pose = PoseStamped()
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            path.poses.insert(0, pose)
            current = came_from[current]

        return path
    

    def calculate_adaptive_lookahead(self, current_pose):
        """ Calculate adaptive lookahead distance based on robot speed and path curvature. """
        base_lookahead = self.lookahead_distance
        
        # Adjust based on current speed (if we had speed feedback)
        # For now, use base velocity
        speed_factor = min(self.linear_velocity / 0.1, 2.0)  # Normalize to 0.1 m/s
        base_lookahead *= speed_factor
        
        # Adjust based on path curvature if available
        if self.current_path and len(self.current_path.poses) > 2:
            # Get current position in path
            robot_x = current_pose.position.x
            robot_y = current_pose.position.y
            
            # Find closest point
            min_dist = float('inf')
            closest_idx = 0
            for i, pose_stamped in enumerate(self.current_path.poses):
                px = pose_stamped.pose.position.x
                py = pose_stamped.pose.position.y
                dist = np.sqrt((robot_x - px)**2 + (robot_y - py)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            # Calculate local curvature
            if closest_idx < len(self.current_path.poses) - 2:
                p1 = self.current_path.poses[closest_idx].pose.position
                p2 = self.current_path.poses[closest_idx + 1].pose.position
                p3 = self.current_path.poses[closest_idx + 2].pose.position
                
                # Vector from p1 to p2 and p2 to p3
                v1 = np.array([p2.x - p1.x, p2.y - p1.y])
                v2 = np.array([p3.x - p2.x, p3.y - p2.y])
                
                # Normalize vectors
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                if v1_norm > 0.001 and v2_norm > 0.001:
                    v1_unit = v1 / v1_norm
                    v2_unit = v2 / v2_norm
                    
                    # Curvature is related to the angle between vectors
                    dot_product = np.dot(v1_unit, v2_unit)
                    dot_product = np.clip(dot_product, -1.0, 1.0)  # Avoid numerical issues
                    angle = np.arccos(dot_product)
                    
                    # Adjust lookahead based on curvature
                    if angle > 0.5:  # High curvature (sharp turn)
                        base_lookahead *= 0.5
                    elif angle < 0.1:  # Low curvature (straight)
                        base_lookahead *= 1.5
        
        # Clamp to valid range
        return np.clip(base_lookahead, self.min_lookahead_distance, self.max_lookahead_distance)

    def calculate_target_orientation(self, current_pose):
        """ Calculate the target orientation based on the first segment of the path. """
        if not self.current_path or len(self.current_path.poses) < 2:
            return None
        
        # Get current robot position
        robot_x = current_pose.position.x
        robot_y = current_pose.position.y
        
        # Find the first target point (could be the first point or a point ahead)
        # For initial positioning, we'll use the first point in the path
        target_pose = self.current_path.poses[0].pose
        target_x = target_pose.position.x
        target_y = target_pose.position.y
        
        # Calculate the angle from robot to target
        dx = target_x - robot_x
        dy = target_y - robot_y
        
        # Calculate the target orientation
        target_angle = np.arctan2(dy, dx)
        
        # Normalize angle to [-pi, pi]
        target_angle = self.angle_diff(target_angle, 0)
        
        return target_angle

    def get_lookahead_point(self, current_pose):
        """ Get the lookahead point for Pure Pursuit algorithm. """
        if not self.current_path or not self.current_path.poses:
            return None
        
        robot_x = current_pose.position.x
        robot_y = current_pose.position.y
        
        # Find the closest point on the path
        min_dist = float('inf')
        closest_idx = 0
        
        for i, pose_stamped in enumerate(self.current_path.poses):
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            dist = np.sqrt((robot_x - px)**2 + (robot_y - py)**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Look ahead from the closest point
        lookahead_dist = self.calculate_adaptive_lookahead(current_pose)
        
        # Find the point at lookahead distance
        cumulative_dist = 0.0
        for i in range(closest_idx, len(self.current_path.poses)):
            if i == closest_idx:
                continue
                
            prev_pose = self.current_path.poses[i-1].pose
            curr_pose = self.current_path.poses[i].pose
            
            segment_length = np.sqrt(
                (curr_pose.position.x - prev_pose.position.x)**2 +
                (curr_pose.position.y - prev_pose.position.y)**2
            )
            
            cumulative_dist += segment_length
            
            if cumulative_dist >= lookahead_dist:
                return curr_pose
        
        # If no point found at lookahead distance, return the last point
        return self.current_path.poses[-1].pose
    
    def compute_control(self, current_pose, target_pose):
        """ Compute control using Pure Pursuit algorithm. """
        # 1) Calcula la posición del lookahead en coords del mapa:
        x = current_pose.position.x
        y = current_pose.position.y
        dx_map = target_pose.position.x - x
        dy_map = target_pose.position.y - y
        
        # 2) Transforma al sistema del robot (rotación inversa de yaw):
        q = current_pose.orientation
        robot_yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('zyx')[0]
        x_r = math.cos(robot_yaw) * dx_map + math.sin(robot_yaw) * dy_map
        y_r = -math.sin(robot_yaw) * dx_map + math.cos(robot_yaw) * dy_map

        # 3) Pure-Pursuit: curvatura
        L = self.lookahead_distance
        if abs(L) < 1e-6:
            curvature = 0.0
        else:
            curvature = 2.0 * y_r / (L * L)

        # 4) Genera el Twist
        cmd = Twist()
        cmd.linear.x = self.linear_velocity        # v_max
        cmd.angular.z = self.linear_velocity * curvature

        # 5) Limitación de curvatura 
        if abs(cmd.linear.x) > 1e-3:  
            curvature = abs(cmd.angular.z / cmd.linear.x)
            if curvature > self.max_curvature:
                
                scale_factor = self.max_curvature / curvature
                cmd.linear.x *= scale_factor
            

            cmd.linear.x = max(cmd.linear.x, 0.05)

        return cmd
    

    def detect_obstacle(self):
        """ Check if there is an obstacle in the way using the latest scan data. """
        if self.latest_scan is None:
            return False

        # Get current robot orientation
        pose = self.estimate_pose()
        q = pose.orientation
        _, _, robot_yaw = quat2euler([q.w, q.x, q.y, q.z])
        
        # Define the forward direction cone (e.g., ±30 degrees from robot's forward direction)
        forward_cone_angle = np.pi/4  # 45 degrees
        
        # Log the cone extremes
        cone_start = robot_yaw - forward_cone_angle
        cone_end = robot_yaw + forward_cone_angle
        self.get_logger().info(f'Obstacle detection cone: {np.degrees(cone_start):.2f}° to {np.degrees(cone_end):.2f}° (robot yaw: {np.degrees(robot_yaw):.2f}°)')
        
        # Count scans in cone range
        scans_in_cone = 0
        
        for i, r in enumerate(self.latest_scan.ranges):
            # Calculate the angle of this laser reading
            angle = self.latest_scan.angle_min + self.latest_scan.angle_increment * i
            laser_angle_in_map = robot_yaw + angle

            # Check if this reading is in the forward direction cone
            # angle_diff = abs(self.angle_diff(angle, robot_yaw)) #loggear esto
            angle_diff = abs(self.angle_diff(laser_angle_in_map, robot_yaw))

            if angle_diff <= forward_cone_angle:
                scans_in_cone += 1
                
            if angle_diff <= forward_cone_angle and r < self.obstacle_detection_distance:
                self.object_avoidance_direction = np.random.choice([-1, 1])
                self.get_logger().info(f'Obstacle detected at angle: {np.degrees(laser_angle_in_map):.1f}°, forward_cone_angle: {forward_cone_angle:.1f}°, distance {r:.2f}m, angle_min: {self.latest_scan.angle_min:.1f}° object_avoidance_direction: {self.object_avoidance_direction}')
                self.stop_robot()
                return True
        
        # Log the number of scans in cone
        self.get_logger().info(f'Scans in cone range: {scans_in_cone} out of {len(self.latest_scan.ranges)} total scans')
                
        return False


    def inflate_map(self):
        """
        Inflates the occupancy grid map by marking cells within a certain radius
        of an obstacle as occupied. This adds a safety buffer around obstacles.
        """
        if self.map_data is None:
            return

        width = self.map_data.info.width
        height = self.map_data.info.height
        resolution = self.map_data.info.resolution
        grid = np.array(self.map_data.data).reshape((height, width))

        inflated_grid = np.copy(grid)
        radius = self.safety_margin_cells

        for y in range(height):
            for x in range(width):
                if grid[y, x] > 50:  # obstacle cell
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                inflated_grid[ny, nx] = 100  # mark as obstacle

        self.grid = inflated_grid
        self.inflated_grid = inflated_grid
    
    def precompute_distance_map(self):
        """
        Pre-computes the distance transform of the map.
        This creates a map where each cell's value is the distance to the
        nearest obstacle. This is done once to speed up the measurement model.
        """
        if self.grid is None:
            self.get_logger().warn("Grid map is not available for distance map pre-computation.")
            return

        self.get_logger().info("Pre-computing distance map...")
        
        # We need to import this, but let's do it locally in case scipy is not a hard dependency
        from scipy.ndimage import distance_transform_edt

        # The distance_transform_edt function calculates the distance to the nearest
        # background element (0). In our grid, obstacles are > 50 and free space is 0.
        # So, we want to find the distance to the nearest non-zero (obstacle) cell.
        # We create a binary grid where obstacles are 0 and free space is 1.
        binary_grid = self.grid < 50
        
        # The result is the Euclidean distance to the nearest 0-cell (obstacle)
        self.distance_map = distance_transform_edt(binary_grid) * self.map_data.info.resolution
        self.get_logger().info("Distance map pre-computed successfully.")

    def stop_robot(self):
        """ Stop the robot by publishing a zero velocity command. """
        self.get_logger().info('Stopping robot due to no odom transform available.')
        self.state = State.IDLE
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

    def timer_callback(self):
        """ Timer callback to run the localization and navigation logic. 
        This is called periodically to update the robot's state and pose estimation. """
        # OK: Implementar maquina de estados para cada caso.
        # Debe haber estado para PLANNING, NAVIGATING y AVOIDING_OBSTACLE, pero pueden haber más estados si se desea.
        if not self.map_received:
            return

        # --- Localization (always running) ---
        if self.latest_scan is None:
            return

        if not self.initial_pose_received:
            self.initialize_particles_randomly()
            self.initial_pose_received = True
            return

        current_odom_tf = self.get_odom_transform()
        if current_odom_tf is None:
            if self.state in [State.NAVIGATING, State.AVOIDING_OBSTACLE, State.INITIAL_POSITIONING]:
                self.stop_robot()
            return

        self.motion_model(current_odom_tf)
        self.measurement_model()
        self.resample()
        estimated_pose = self.estimate_pose()
        self.publish_pose(estimated_pose)
        self.publish_particles()
        self.publish_transform(estimated_pose, current_odom_tf)

        if self.state == State.IDLE:
            self.stop_robot()
            return
        elif self.state == State.PLANNING:
            if self.goal_pose is None:
                self.get_logger().warn("Goal pose not set. Cannot plan path.")
                self.state = State.IDLE
                return
            
            start = self.world_to_grid(estimated_pose.position.x, estimated_pose.position.y)
            goal = self.world_to_grid(self.goal_pose.position.x, self.goal_pose.position.y)
            path = self.plan_path(start, goal)

            if not path or len(path.poses) == 0:
                self.get_logger().warn("No valid path found from current pose to goal.")
                self.state = State.IDLE
                self.stop_robot()
                return
            
            self.current_path = path
            self.publish_path(path)
            
            # Calculate target orientation for initial positioning
            self.target_orientation = self.calculate_target_orientation(estimated_pose)
            if self.target_orientation is not None:
                self.get_logger().info(f"Path planned. Starting initial positioning to target orientation: {np.degrees(self.target_orientation):.2f}°")
                self.state = State.INITIAL_POSITIONING
            else:
                self.get_logger().warn("Could not calculate target orientation. Starting navigation directly.")
                self.state = State.NAVIGATING
            return
            
        elif self.state == State.INITIAL_POSITIONING:
            if self.target_orientation is None:
                self.get_logger().warn("Target orientation not set. Switching to NAVIGATING.")
                self.state = State.NAVIGATING
                return
            
            # Get current robot orientation
            q = estimated_pose.orientation
            _, _, current_yaw = quat2euler([q.w, q.x, q.y, q.z])
            
            # Calculate angle difference
            angle_diff = self.angle_diff(self.target_orientation, current_yaw)
            
            # Check if we're close enough to target orientation
            if abs(angle_diff) < self.initial_positioning_tolerance:
                self.get_logger().info(f"Initial positioning complete. Current yaw: {np.degrees(current_yaw):.2f}°, Target: {np.degrees(self.target_orientation):.2f}°")
                self.state = State.NAVIGATING
                self.target_orientation = None
                return
            
            # Rotate towards target orientation
            twist = Twist()
            twist.angular.z = self.initial_positioning_angular_speed * np.sign(angle_diff)
            self.cmd_vel_pub.publish(twist)
            
            self.get_logger().info(f"Initial positioning - Current: {np.degrees(current_yaw):.2f}°, Target: {np.degrees(self.target_orientation):.2f}°, Diff: {np.degrees(angle_diff):.2f}°")
            return
            
        elif self.state == State.NAVIGATING:
            if self.detect_obstacle():
                self.get_logger().info("Obstacle detected. Switching to AVOIDING_OBSTACLE state.")
                self.obstacle_avoidance_cumulative_angle = 0.0
                _, _, self.obstacle_avoidance_start_yaw = quat2euler([
                    estimated_pose.orientation.w,
                    estimated_pose.orientation.x,
                    estimated_pose.orientation.y,
                    estimated_pose.orientation.z
                ])
                self.obstacle_avoidance_last_yaw = self.obstacle_avoidance_start_yaw
                self.state = State.AVOIDING_OBSTACLE
                return

            dist_to_goal = np.hypot(
                estimated_pose.position.x - self.goal_pose.position.x,
                estimated_pose.position.y - self.goal_pose.position.y) # hypot es como hacer sqrt(x**2 + y**2)

            if dist_to_goal < self.goal_tolerance:
                self.get_logger().info("Llegamos locoo!") 
                self.stop_robot()
                self.state = State.IDLE
            else:
                lookahead = self.get_lookahead_point(estimated_pose)
                if lookahead is not None:
                    cmd = self.compute_control(estimated_pose, lookahead)
                    self.cmd_vel_pub.publish(cmd)
                else:
                    self.get_logger().warn("No lookahead point found. Stopping robot.")
                    self.stop_robot()
                    self.state = State.IDLE
        
        elif self.state == State.AVOIDING_OBSTACLE:
            _, _, current_yaw = quat2euler([
                estimated_pose.orientation.w,
                estimated_pose.orientation.x,
                estimated_pose.orientation.y,
                estimated_pose.orientation.z
            ])
            
            yaw_diff = self.angle_diff(current_yaw, self.obstacle_avoidance_last_yaw)
            self.obstacle_avoidance_cumulative_angle += abs(yaw_diff)
            self.obstacle_avoidance_last_yaw = current_yaw
            
            twist = Twist()
            twist.angular.z = self.obstacle_avoidance_turn_speed * self.object_avoidance_direction
            self.cmd_vel_pub.publish(twist)

            if self.obstacle_avoidance_cumulative_angle > np.pi / 2:
                # 1. Obtener la pose estimada actual
                estimated_pose = self.estimate_pose()
                start = self.world_to_grid(estimated_pose.position.x, estimated_pose.position.y)
                goal = self.world_to_grid(self.goal_pose.position.x, self.goal_pose.position.y)
                
                # 2. Recalcular el path
                new_path = self.plan_path(start, goal)
                if new_path:
                    # 3. Actualizar el path actual
                    self.current_path = new_path
                    # 4. Publicar el nuevo path
                    self.publish_path(self.current_path)
                    # 5. Recalculate target orientation for new path
                    self.target_orientation = self.calculate_target_orientation(estimated_pose)
                    self.get_logger().info('Path replanned after obstacle avoidance.')
                else:
                    self.get_logger().warn('Could not replan path after obstacle avoidance.')
                    # Check if there is still an obstacle in front
                    # For now, just continue with current path
                
                self.get_logger().info("Finished obstacle avoidance maneuver. Resuming navigation.")
                self.state = State.NAVIGATING


    def initial_pose_callback(self, msg):
        if msg.header.frame_id != self.map_frame_id:
            self.get_logger().warn(f"Initial pose frame is '{msg.header.frame_id}' but expected '{self.map_frame_id}'. Ignoring.")
            return
        self.get_logger().info('Initial pose received.')
        self.initialize_particles(msg.pose.pose)
        self.publish_pose(msg.pose.pose)
        odom_tf = self.get_odom_transform()
        if odom_tf:
            self.publish_transform(msg.pose.pose, odom_tf)
        else:
            self.get_logger().warn('Could not get odom transform after initial pose')
        
        # Reset state
        self.state = State.IDLE
        self.initial_pose_received = True
        self.last_odom_pose = None # Reset odom tracking
        self.target_orientation = None  # Reset target orientation
        self.stop_robot()

    def initialize_particles(self, initial_pose):
        x = initial_pose.position.x
        y = initial_pose.position.y
        q = initial_pose.orientation
        theta = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]

        pos_std = 0.5
        theta_std = 0.3

        N = self.num_particles

        self.particles[:, 0] = np.random.normal(x, pos_std, N)     
        self.particles[:, 1] = np.random.normal(y, pos_std, N)     
        self.particles[:, 2] = np.random.normal(theta, theta_std, N)    

        # Normalizamos los ángulos a [-pi, pi]
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

        self.weights = np.ones(N) / N

        self.publish_particles()

    def initialize_particles_randomly(self):
        # OK: Inizializar particulas aleatoriamente en todo el mapa
        free_indices = np.where(np.array(self.map_data.data) == 0)[0]
        if len(free_indices) < self.num_particles:
            self.get_logger().warn("Not enough free space to initialize all particles.")
            return

        indices = np.random.choice(free_indices, self.num_particles, replace=False)
        width = self.map_data.info.width
        
        xs = []
        ys = []
        for idx in indices:
            gy = idx // width  
            gx = idx % width  
            wx, wy = self.grid_to_world(gx, gy)
            xs.append(wx)
            ys.append(wy)

        self.particles[:, 0] = xs
        self.particles[:, 1] = ys
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, self.num_particles)

        self.weights = np.ones(self.num_particles) / self.num_particles

        self.publish_particles()


    def get_odom_transform(self):
        try:
            return self.tf_buffer.lookup_transform(self.odom_frame_id, self.base_frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(f'Could not get transform from {self.odom_frame_id} to {self.base_frame_id}. Skipping update. Error: {e}', throttle_duration_sec=2.0)
            return None

    def motion_model(self, current_odom_tf):
        current_odom_pose = current_odom_tf.transform

        x_odom, y_odom = current_odom_pose.translation.x, current_odom_pose.translation.y
        orientation = current_odom_pose.rotation
        quaternion = [orientation.w, orientation.x, orientation.y, orientation.z]  # ¡Atención! orden: w, x, y, z
        _, _, theta_odom = quat2euler(quaternion) 

        if self.last_odom_pose is not None:
            t0 = self.last_odom_pose.translation
            r0 = self.last_odom_pose.rotation
            x_prev, y_prev = t0.x, t0.y
            q_prev = [r0.w, r0.x, r0.y, r0.z]
            _, _, theta_prev = quat2euler(q_prev)

            delta_theta = self.angle_diff(theta_odom, theta_prev)
            delta_trans = np.sqrt((x_odom - x_prev) ** 2 + (y_odom - y_prev) ** 2)
            delta_rot1 = np.arctan2(y_odom - y_prev, x_odom - x_prev) - theta_prev
            delta_rot2 = delta_theta - delta_rot1

            delta_rot1 = self.angle_diff(delta_rot1, 0)
            delta_rot2 = self.angle_diff(delta_rot2, 0)

            if delta_trans > 0.01 or abs(delta_theta) > 0.01:
                for i in range(self.num_particles):
                    p = self.particles[i]
                    delta_rot1_hat = delta_rot1 + np.random.normal(0, self.alphas[0] * abs(delta_rot1) + self.alphas[1] * delta_trans)
                    delta_trans_hat = delta_trans + np.random.normal(0, self.alphas[2] * delta_trans + self.alphas[3] * (abs(delta_rot1) + abs(delta_rot2)))
                    delta_rot2_hat = delta_rot2 + np.random.normal(0, self.alphas[0] * abs(delta_rot2) + self.alphas[1] * delta_trans)

                    p[0] += delta_trans_hat * np.cos(p[2] + delta_rot1_hat)
                    p[1] += delta_trans_hat * np.sin(p[2] + delta_rot1_hat)
                    p[2] += delta_rot1_hat + delta_rot2_hat
                    p[2] = self.angle_diff(p[2], 0)
                    self.particles[i] = p
            else:
                for i in range(self.num_particles):
                    p = self.particles[i]
                    noise_theta = np.random.normal(0, self.alphas[0] * abs(delta_theta))
                    p[2] += delta_theta + noise_theta
                    p[2] = self.angle_diff(p[2], 0)
                    self.particles[i] = p

        # Clamp particles to map boundaries
        min_x = self.map_data.info.origin.position.x
        min_y = self.map_data.info.origin.position.y
        max_x = min_x + self.map_data.info.width * self.map_data.info.resolution
        max_y = min_y + self.map_data.info.height * self.map_data.info.resolution

        self.particles[:, 0] = np.clip(self.particles[:, 0], min_x, max_x)
        self.particles[:, 1] = np.clip(self.particles[:, 1], min_y, max_y)

        self.last_odom_pose = current_odom_pose

    def measurement_model(self):
        if self.latest_scan is None:
            return

        if self.distance_map is None:
            self.get_logger().warn("Distance map not available, skipping measurement model.")
            return

        map_w = self.map_data.info.width
        map_h = self.map_data.info.height

        # Process each particle
        for i, (x, y, theta) in enumerate(self.particles):
            score = 1.0
            
            # Get real scan rays
            angles = [theta + self.latest_scan.angle_min + j * self.latest_scan.angle_increment 
                     for j in range(len(self.latest_scan.ranges))]
            
            # Sample some rays to reduce computation (every 5 ray) # Changed to 2
            for j in range(0, len(self.latest_scan.ranges), 2):
                measured_range = self.latest_scan.ranges[j]
                
                # Skip invalid measurements
                if measured_range >= self.latest_scan.range_max or measured_range <= self.latest_scan.range_min:
                    continue
                
                # Calculate expected beam endpoint
                beam_x = x + measured_range * np.cos(angles[j])
                beam_y = y + measured_range * np.sin(angles[j])
                
                # Convert to grid coordinates
                gx, gy = self.world_to_grid(beam_x, beam_y)
                
                if 0 <= gx < map_w and 0 <= gy < map_h:
                    # Get distance to nearest obstacle from pre-calculated map
                    dist = self.distance_map[gy, gx]
                    
                    # Calculate likelihood using hit model and random noise
                    # Using Gaussian for hit model
                    sigma = 0.2  # Standard deviation for hit model
                    p_hit = np.exp(-dist**2 / (2 * sigma**2))
                    p = self.z_hit * p_hit + self.z_rand
                    
                    score *= p

            self.weights[i] = score

        # Normalize weights
        max_weight = max(self.weights)
        if max_weight > 0:
            # Normalize relative to max weight to avoid numerical issues
            self.weights = self.weights / max_weight
            # Then normalize to sum to 1
            self.weights = self.weights / np.sum(self.weights)
        else:
            # If all weights are zero, use uniform distribution
            self.weights = np.ones(self.num_particles) / self.num_particles


    
    def resample(self):
        # OK: Implementar el resampleo de las particulas basado en los pesos.
        new_particles = []
        
        # Check if resampling is needed
        max_weight = max(self.weights)
        effective_particles = sum(1 for w in self.weights if w > max_weight * 0.5)
        
        # Normalize weights
        total_weight = sum(self.weights)
        if total_weight <= 0:
            # If all weights are zero, reinitialize particles randomly
            self.initialize_particles_randomly()
            return
            
        normalized_weights = [w / total_weight for w in self.weights]
        
        # Systematic resampling
        r = np.random.uniform(0, 1.0 / self.num_particles)
        c = normalized_weights[0]
        i = 0
        
        for m in range(self.num_particles):
            u = r + m / self.num_particles
            while u > c and i < len(normalized_weights) - 1:
                i += 1
                c += normalized_weights[i]
            if i < len(self.particles):
                p = self.particles[i]
                new_particles.append([p[0], p[1], p[2]])
        
        self.particles = np.array(new_particles)
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Add random particles if diversity is low
        if effective_particles < self.num_particles * 0.3:
            self.add_random_particles(0.2)
        elif effective_particles < self.num_particles * 0.5:
            self.add_random_particles(0.05)

    def add_random_particles(self, fraction):
        num_random = int(len(self.particles) * fraction)
        
        # Get top 3 particles to improve diversity
        top_indices = np.argsort(self.weights)[-3:]
        
        for i in range(num_random):
            if i < len(self.particles):
                # Randomly choose from top particles
                chosen_idx = np.random.choice(top_indices)
                chosen_particle = self.particles[chosen_idx]
                
                # Add noise with adaptive scale based on particle weight
                weight_factor = self.weights[chosen_idx] / max(self.weights)
                noise_scale = 0.1 + 0.2 * (1 - weight_factor)  # More noise for lower weight particles
                
                noise_x = np.random.normal(0, noise_scale)
                noise_y = np.random.normal(0, noise_scale)
                noise_theta = np.random.normal(0, 0.1)
                
                self.particles[i][0] = chosen_particle[0] + noise_x
                self.particles[i][1] = chosen_particle[1] + noise_y
                self.particles[i][2] = self.angle_diff(chosen_particle[2] + noise_theta, 0)


    def estimate_pose(self):
        # OK: Implementar la estimación de pose a partir de las particulas y sus pesos.
        x = np.average(self.particles[:, 0], weights=self.weights)
        y = np.average(self.particles[:, 1], weights=self.weights)

        sin_weighted = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
        cos_weighted = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
        theta = np.arctan2(sin_weighted, cos_weighted)

        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        q = R.from_euler('z', theta).as_quat()
        pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        return pose

    def publish_pose(self, estimated_pose):
        p = PoseWithCovarianceStamped()
        p.header.stamp = self.get_clock().now().to_msg()
        p.header.frame_id = self.map_frame_id
        p.pose.pose = estimated_pose
        self.pose_pub.publish(p)


    def publish_particles(self):
        ma = MarkerArray()
        for i, p in enumerate(self.particles):
            marker = Marker()
            marker.header.frame_id = self.map_frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "particles"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = p[0]
            marker.pose.position.y = p[1]
            q = euler2quat(0, 0, p[2])  # Convertimos el ángulo a quaternion
            marker.pose.orientation = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
            marker.scale.x = 0.1
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.color.a = 0.5
            marker.color.r = 1.0
            ma.markers.append(marker)
        self.particle_pub.publish(ma)

    def publish_transform(self, estimated_pose, odom_tf):
        map_to_base_mat = self.pose_to_matrix(estimated_pose)
        odom_to_base_mat = self.transform_to_matrix(odom_tf.transform)
        map_to_odom_mat = np.dot(map_to_base_mat, np.linalg.inv(odom_to_base_mat))
        
        t = TransformStamped()
        
        # OK: Completar el TransformStamped con la transformacion entre el mapa y la base del robot.
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame_id
        t.child_frame_id = self.odom_frame_id
        t.transform.translation.x = map_to_odom_mat[0, 3]
        t.transform.translation.y = map_to_odom_mat[1, 3]
        t.transform.translation.z = map_to_odom_mat[2, 3]
        rot = R.from_matrix(map_to_odom_mat[:3, :3]).as_quat()
        t.transform.rotation = Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])

        self.tf_broadcaster.sendTransform(t)

    def pose_to_matrix(self, pose):
        q = pose.orientation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        mat = np.eye(4)
        mat[:3, :3] = r.as_matrix()
        mat[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return mat

    def transform_to_matrix(self, transform):
        q = transform.rotation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        mat = np.eye(4)
        mat[:3, :3] = r.as_matrix()
        t = transform.translation
        mat[:3, 3] = [t.x, t.y, t.z]
        return mat

    def world_to_grid(self, wx, wy):
        gx = int((wx - self.map_data.info.origin.position.x) / self.map_data.info.resolution)
        gy = int((wy - self.map_data.info.origin.position.y) / self.map_data.info.resolution)
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        wx = gx * self.map_data.info.resolution + self.map_data.info.origin.position.x
        wy = gy * self.map_data.info.resolution + self.map_data.info.origin.position.y
        return (wx, wy)
    

    def publish_path(self, path_msg):
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame_id
        self.path_pub.publish(path_msg)

    def prune_path(self, current_pose):
        if self.current_path is None or len(self.current_path.poses) < 2:
            return

        robot_x = current_pose.position.x
        robot_y = current_pose.position.y

        # Buscar el índice del punto más cercano
        min_dist_sq = float('inf')
        closest_idx = -1
        for i, pose_stamped in enumerate(self.current_path.poses):
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            dist_sq = (robot_x - px)**2 + (robot_y - py)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_idx = i

        # Si está suficientemente cerca y no es el primero, recortar el camino
        if closest_idx > 0 and math.sqrt(min_dist_sq) < self.path_pruning_distance:
            self.current_path.poses = self.current_path.poses[closest_idx:]

    @staticmethod
    def angle_diff(a, b):
        d = a - b
        while d > np.pi:
            d -= 2 * np.pi
        while d < -np.pi:
            d += 2 * np.pi
        return d


def main(args=None):
    rclpy.init(args=args)
    node = AmclNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()