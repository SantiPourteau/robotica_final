#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
# import tf_transformations
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import transforms3d as tf3d 
from scipy.spatial.transform import Rotation as R


class Particle:
    def __init__(self, x, y, theta, weight, map_shape):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
        self.log_odds_map = np.zeros(map_shape, dtype=np.float32)
        self.previous_odom_pose = [0.0, 0.0, 0.0]

    def pose(self):
        return np.array([self.x, self.y, self.theta])

class PythonSlamNode(Node):
    def __init__(self):
        super().__init__('python_slam_node')

        # Parameters
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')

        # TODO: define map resolution, width, height, and number of particles
        self.declare_parameter('map_resolution', 0.05)
        self.declare_parameter('map_width_meters', 6.0)
        self.declare_parameter('map_height_meters', 6.0)
        self.declare_parameter('num_particles', 10)


        self.resolution = self.get_parameter('map_resolution').get_parameter_value().double_value
        self.map_width_m = self.get_parameter('map_width_meters').get_parameter_value().double_value
        self.map_height_m = self.get_parameter('map_height_meters').get_parameter_value().double_value
        self.map_width_cells = int(self.map_width_m / self.resolution)
        self.map_height_cells = int(self.map_height_m / self.resolution)
        self.map_origin_x = -self.map_width_m / 2.0
        self.map_origin_y = -5.0

        # TODO: define the log-odds criteria for free and occupied cells
        self.log_odds_free = -0.1
        self.log_odds_occupied = 0.2
        self.log_odds_max = 10.0
        self.log_odds_min = -10.0


        # Particle filter
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.particles = [Particle(0.0, 0.0, 0.0, 1.0/self.num_particles, (self.map_height_cells, self.map_width_cells)) for _ in range(self.num_particles)]
        self.last_odom = None
        self.current_map_pose = [0.0, 0.0, 0.0]
        self.current_odom_pose = [0.0, 0.0, 0.0]

        # Parametros Agregados
        self.max_lidar_distance = 3.0

        # ROS2 publishers/subscribers
        map_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', map_qos_profile)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_subscriber = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').get_parameter_value().string_value,
            self.odom_callback,
            10)
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').get_parameter_value().string_value,
            self.scan_callback,
            rclpy.qos.qos_profile_sensor_data)

        self.get_logger().info("Python SLAM node with particle filter initialized.")
        self.map_publish_timer = self.create_timer(1.0, self.publish_map)

        # Inicializa el buffer y el listener de TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def odom_callback(self, msg: Odometry):
        # Store odometry for motion update
        self.last_odom = msg

    def scan_callback(self, msg: LaserScan):
        if self.last_odom is None:
            return

        # Filtra los rebotes de los sensores
        rangos_filtrados = []
        for r in msg.ranges:
            if math.isnan(r) or r < msg.range_min or r >= self.max_lidar_distance:
                rangos_filtrados.append(float('nan'))
            else:
                rangos_filtrados.append(r)
        # Sobrescribimos msg.ranges para que compute_weight() y update_map() los ignoren
        msg.ranges = rangos_filtrados


        # 1. Motion update (sample motion model)
        odom = self.last_odom
        # TODO: Retrieve odom_pose from odom message - remember that orientation is a quaternion

        odom_x = odom.pose.pose.position.x
        odom_y = odom.pose.pose.position.y
        quat = odom.pose.pose.orientation
        roll, pitch, yaw = tf3d.euler.quat2euler([quat.w, quat.x, quat.y, quat.z])
        odom_theta = yaw
        self.current_odom_pose = [odom_x, odom_y, odom_theta]




        alpha1, alpha2, alpha3, alpha4 = 0.005, 0.005, 0.005, 0.005
        # TODO: Model the particles around the current pose
        for p in self.particles:
            # Add noise to simulate motion uncertainty
            dx = odom_x - p.previous_odom_pose[0]
            dy = odom_y - p.previous_odom_pose[1]
            d_theta = self.angle_diff(odom_theta, p.previous_odom_pose[2])

            ruido_x = np.random.normal(0, alpha3 * abs(dx) + alpha4 * abs(d_theta))
            ruido_y = np.random.normal(0, alpha3 * abs(dy) + alpha4 * abs(d_theta))
            ruido_theta = np.random.normal(0, alpha1 * abs(d_theta) + alpha2 * (abs(dx) + abs(dy)))
            
            p.x += dx + ruido_x
            p.y += dy + ruido_y
            p.theta = self.angle_diff(p.theta + d_theta + ruido_theta,0)

            p.previous_odom_pose = [odom_x, odom_y, odom_theta]


        # TODO: 2. Measurement update (weight particles)
        weights = []
        for p in self.particles:
            weight = self.compute_weight(p, msg) # Compute weights for each particle
            # Save, append
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(self.particles) for _ in range(len(self.particles))]

        for i, p in enumerate(self.particles):
            p.weight = weights[i] # Resave weights

        # 3. Resample
        self.particles = self.resample_particles(self.particles)

        # TODO: 4. Use weighted mean of all particles for mapping and pose (update current_map_pose and current_odom_pose, for each particle)
        x_pesado = sum(p.x * p.weight for p in self.particles)
        y_pesado = sum(p.y * p.weight for p in self.particles)
        theta_pesado = sum(p.theta * p.weight for p in self.particles)
        self.current_map_pose = [x_pesado, y_pesado, theta_pesado]


        # 5. Mapping (update map with best particle's pose)
        for p in self.particles:
            self.update_map(p, msg)

        # 6. Broadcast map->odom transform
        self.broadcast_map_to_odom()

    def compute_weight(self, particle, scan_msg):
        # Simple likelihood: count how many endpoints match occupied cells
        score = 0.0
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        for i, range_dist in enumerate(scan_msg.ranges):
            if range_dist < scan_msg.range_min or range_dist > scan_msg.range_max or math.isnan(range_dist):
                continue
            # TODO: Compute the map coordinates of the endpoint: transform the scan into the map frame
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            endpoint_x = robot_x + range_dist * math.cos(robot_theta + angle)
            endpoint_y = robot_y + range_dist * math.sin(robot_theta + angle)
            cell_x = int((endpoint_x - self.map_origin_x) / self.resolution)
            cell_y = int((endpoint_y - self.map_origin_y) / self.resolution)
            if 0 <= cell_x < self.map_width_cells and 0 <= cell_y < self.map_height_cells:
                # TODO: Use particle.log_odds_map for scoring
                if particle.log_odds_map[cell_y, cell_x] > 0:
                    score += 1.0
        return score + 1e-6


    def resample_particles(self, particles):
        # TODO: Resample particles
        #Implementar metodo de resampling Stochastic Universal Sampling (SUS)
        new_particles = []
        weights = [p.weight for p in particles]
        total_weight = sum(weights)
        if total_weight <= 0:
            for p in particles:
                p.weight = 1.0 / len(particles)
            return particles
        weights = [w / total_weight for w in weights]
        cum_weights = np.cumsum(weights)
        step = 1.0 / len(particles)
        start = np.random.uniform(0, step)
        pointers = [start + i * step for i in range(len(particles))]
        for pointer in pointers:
            idx = np.searchsorted(cum_weights, pointer)
            if idx >= len(particles):
                idx = len(particles) - 1 
            p = particles[idx]
            new_particle = Particle(p.x, p.y, p.theta, 1.0/len(particles), (self.map_height_cells, self.map_width_cells))
            new_particle.log_odds_map = p.log_odds_map.copy()
            new_particle.previous_odom_pose = p.previous_odom_pose.copy()
            new_particles.append(new_particle)
        return new_particles

    def update_map(self, particle, scan_msg):
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        for i, range_dist in enumerate(scan_msg.ranges):
            is_hit = range_dist < self.max_lidar_distance
            current_range = min(range_dist, self.max_lidar_distance)
            if math.isnan(current_range) or current_range >= self.max_lidar_distance or current_range < scan_msg.range_min:
                continue

            #Problema con las esquinas de las paredes
            if range_dist > 0.9 * self.max_lidar_distance:
                continue
            # TODO: Update map: transform the scan into the map frame
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            endpoint_x = robot_x + current_range * math.cos(robot_theta + angle)
            endpoint_y = robot_y + current_range * math.sin(robot_theta + angle)

            endpoint_cell_x = int((endpoint_x - self.map_origin_x) / self.resolution)
            endpoint_cell_y = int((endpoint_y - self.map_origin_y) / self.resolution)

            robot_cell_x = int((robot_x - self.map_origin_x) / self.resolution)
            robot_cell_y = int((robot_y - self.map_origin_y) / self.resolution)

            # TODO: Use self.bresenham_line for free cells
            self.bresenham_line(particle, robot_cell_x, robot_cell_y, endpoint_cell_x, endpoint_cell_y)
            
        
            # TODO: Update particle.log_odds_map accordingly
            if is_hit and 0 <= endpoint_cell_x < self.map_width_cells and 0 <= endpoint_cell_y < self.map_height_cells:
                particle.log_odds_map[endpoint_cell_y, endpoint_cell_x] += self.log_odds_occupied
                particle.log_odds_map[endpoint_cell_y, endpoint_cell_x] = np.clip(particle.log_odds_map[endpoint_cell_y, endpoint_cell_x], self.log_odds_min, self.log_odds_max)

            

    def bresenham_line(self, particle, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        path_len = 0
        max_path_len = dx + dy
        while not (x0 == x1 and y0 == y1) and path_len < max_path_len:
            if 0 <= x0 < self.map_width_cells and 0 <= y0 < self.map_height_cells:
                particle.log_odds_map[y0, x0] += self.log_odds_free
                particle.log_odds_map[y0, x0] = np.clip(particle.log_odds_map[y0, x0], self.log_odds_min, self.log_odds_max)
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
            path_len += 1

    def publish_map(self):
        # TODO: Fill in map_msg fields and publish one map
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value

        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.map_width_cells
        map_msg.info.height = self.map_height_cells
        map_msg.info.origin.position.x = self.map_origin_x
        map_msg.info.origin.position.y = self.map_origin_y
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0

        #Usar la mejor particula para el mapa publicado
        best_particle = max(self.particles, key=lambda p: p.weight)
        log_odds_map = best_particle.log_odds_map

        occupancy_grid = np.full_like(log_odds_map, -1, dtype=np.int8)
        occupancy_grid[log_odds_map > 0.5] = 100
        occupancy_grid[log_odds_map < -0.5] = 0
        #unknown cells
        occupancy_grid[log_odds_map == 0] = -1

        map_msg.data = occupancy_grid.flatten().tolist()


        self.map_publisher.publish(map_msg)
        self.get_logger().debug("Map published.")

    def broadcast_map_to_odom(self):
        try:
            # 1. Obtener la transformación actual odom->base_link desde el buffer TF
            odom_to_base = self.tf_buffer.lookup_transform(
                self.get_parameter('odom_frame').get_parameter_value().string_value,
                self.get_parameter('base_frame').get_parameter_value().string_value,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(f"No se pudo obtener odom->base_link: {e}", throttle_duration_sec=2.0)
            return

        # 2. Construir la pose estimada del robot en el mapa (map->base_link)
        pose_map = np.eye(4)
        pose_map[:2, 3] = self.current_map_pose[:2]
        rot_map = R.from_euler('z', self.current_map_pose[2]).as_matrix()
        pose_map[:3, :3] = rot_map

        # 3. Construir la matriz homogénea de odom->base_link
        t = odom_to_base.transform.translation
        q = odom_to_base.transform.rotation
        rot_odom = R.from_quat([q.x, q.y, q.z, q.w])
        odom_mat = np.eye(4)
        odom_mat[:3, :3] = rot_odom.as_matrix()
        odom_mat[:3, 3] = [t.x, t.y, t.z]

        # 4. Calcular map->odom como map->base_link * inv(odom->base_link)
        map_to_odom = pose_map @ np.linalg.inv(odom_mat)

        # 5. Extraer traslación y rotación
        trans = map_to_odom[:3, 3]
        rot = R.from_matrix(map_to_odom[:3, :3]).as_quat()

        # 6. Publicar la transformación
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value
        tf_msg.child_frame_id = self.get_parameter('odom_frame').get_parameter_value().string_value
        tf_msg.transform.translation.x = trans[0]
        tf_msg.transform.translation.y = trans[1]
        tf_msg.transform.translation.z = trans[2]
        tf_msg.transform.rotation.x = rot[0]
        tf_msg.transform.rotation.y = rot[1]
        tf_msg.transform.rotation.z = rot[2]
        tf_msg.transform.rotation.w = rot[3]

        self.tf_broadcaster.sendTransform(tf_msg)


    @staticmethod
    def angle_diff(a, b):
        d = a - b
        while d > np.pi:
            d -= 2 * np.pi
        while d < -np.pi:
            d += 2 * np.pi
        return d
    
    # Funciones Auxiliares
    
    

def main(args=None):
    rclpy.init(args=args)
    node = PythonSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()