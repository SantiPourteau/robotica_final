#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from transforms3d.euler import quat2euler, euler2quat
# from tf2_ros import transformations as tft
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

class Particle:
    def __init__(self, x, y, theta, weight, map_shape):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
        self.log_odds_map = np.zeros(map_shape, dtype=np.float32)

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

        # OK TODO: define map resolution, width, height, and number of particles
        self.declare_parameter('map_resolution', 0.05)  # metros por celda
        self.declare_parameter('map_width_meters', 20.0)  # ancho en metros
        self.declare_parameter('map_height_meters', 20.0)  # alto en metros
        self.declare_parameter('num_particles', 10)  # número de partículas - solo 10 como solicitado

        self.resolution = self.get_parameter('map_resolution').get_parameter_value().double_value
        self.map_width_m = self.get_parameter('map_width_meters').get_parameter_value().double_value
        self.map_height_m = self.get_parameter('map_height_meters').get_parameter_value().double_value
        self.map_width_cells = int(self.map_width_m / self.resolution)
        self.map_height_cells = int(self.map_height_m / self.resolution)
        self.map_origin_x = -self.map_width_m / 2.0
        self.map_origin_y = -5.0

        # OK TODO: define the log-odds criteria for free and occupied cells
        self.log_odds_occupied = 1.5
        self.log_odds_free = -0.2
        self.log_odds_max = 15
        self.log_odds_min = -5.0
        
        # Add conservative update factors for cells with existing evidence
        self.occupied_update_factor = 0.3  # Reduce update strength for already occupied cells
        self.free_update_factor = 0.3      # Reduce update strength for already free cells
        self.confidence_threshold = 2.0    # Threshold for considering a cell as "confident"
        
        # Asymmetric confidence thresholds - harder to overwrite occupied with free
        self.occupied_confidence_threshold = 1.5  # Lower threshold for occupied cells
        self.free_confidence_threshold = 3.0      # Higher threshold for free cells

        # OK TODO: define lidar distance limits for useful range
        self.declare_parameter('lidar_min_distance', 0.1)  # distancia mínima útil en metros
        self.declare_parameter('lidar_max_distance', 2.0)  # distancia máxima útil en metros
        
        self.lidar_min_distance = self.get_parameter('lidar_min_distance').get_parameter_value().double_value
        self.lidar_max_distance = self.get_parameter('lidar_max_distance').get_parameter_value().double_value


        self.obstacle_avoidance_forward_distance = 0.0
        self.obstacle_avoidance_forward_done = False

        
        # Particle filter
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.particles = [Particle(0.0, 0.0, 0.0, 1.0/self.num_particles, (self.map_height_cells, self.map_width_cells)) for _ in range(self.num_particles)]
        self.last_odom = None
        self.prev_odom = None
        self.current_map_pose = [0.0, 0.0, 0.0]
        self.current_odom_pose = [0.0, 0.0, 0.0]  # Initialize current_odom_pose
        

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
        self.get_logger().info(f"Map size: {self.map_width_cells}x{self.map_height_cells} cells, resolution: {self.resolution}m")
        self.get_logger().info(f"Number of particles: {self.num_particles}")
        self.get_logger().info(f"Lidar distance limits: {self.lidar_min_distance}m - {self.lidar_max_distance}m")
        self.map_publish_timer = self.create_timer(1.0, self.publish_map)

    def odom_callback(self, msg: Odometry):
        # Store odometry for motion update
        self.get_logger().info(f"Odom callback: {msg}")
        self.prev_odom = self.last_odom
        self.last_odom = msg
        

    def scan_callback(self, msg: LaserScan):

        if self.last_odom is None:
            return

        # 1. Motion update (sample motion model)
        odom = self.last_odom

        # OK TODO: Retrieve odom_pose from odom message - remember that orientation is a quaternion
        
        x_odom, y_odom = odom.pose.pose.position.x, odom.pose.pose.position.y
        orientation = odom.pose.pose.orientation
        quaternion = [orientation.w, orientation.x, orientation.y, orientation.z]  # ¡Atención! orden: w, x, y, z
        _, _, theta_odom = quat2euler(quaternion)

        odom_pose = [x_odom, y_odom, theta_odom] 

        # Initialize motion model variables
        delta_rot1 = 0.0
        delta_trans = 0.0  
        delta_rot2 = 0.0
        # Aumentar ruido para compensar pocas partículas y drift de odometría
        alpha1, alpha2, alpha3, alpha4 = 0.02, 0.02, 0.05, 0.02  # Valores más altos para 10 partículas

        if self.prev_odom is not None:
            prev_position = self.prev_odom.pose.pose.position
            x_prev, y_prev = prev_position.x, prev_position.y
            prev_orientation = self.prev_odom.pose.pose.orientation
            prev_quaternion = [prev_orientation.w, prev_orientation.x, prev_orientation.y, prev_orientation.z]  # ¡Atención! orden: w, x, y, z
            _, _, theta_prev = quat2euler(prev_quaternion)
        
        
            delta_theta = self.angle_diff(theta_odom, theta_prev) 

            # === Calcular los parámetros de odometría ===
            delta_trans = np.sqrt((x_odom - x_prev)**2 + (y_odom - y_prev)**2)
            delta_rot1 = np.arctan2(y_odom - y_prev, x_odom - x_prev) - theta_prev
            delta_rot2 = delta_theta - delta_rot1

            # Normalizar ángulos
            delta_rot1 = self.angle_diff(delta_rot1, 0)
            delta_rot2 = self.angle_diff(delta_rot2, 0)

            # Only update particles if movement exceeds threshold - más sensible para 10 partículas
            if delta_trans > 0.005 or abs(delta_theta) > 0.005:  # 5mm = 0.005m, más sensible
                # === Actualizamos cada partícula usando el modelo de movimiento ===
                for p in self.particles:
                    # Agregar ruido según el modelo probabilístico
                    delta_rot1_hat = delta_rot1 + np.random.normal(0, alpha1 * abs(delta_rot1) + alpha2 * delta_trans)
                    delta_trans_hat = delta_trans + np.random.normal(0, alpha3 * delta_trans + alpha4 * (abs(delta_rot1) + abs(delta_rot2)))
                    delta_rot2_hat = delta_rot2 + np.random.normal(0, alpha1 * abs(delta_rot2) + alpha2 * delta_trans)

                    # Propagar la partícula
                    p.x += delta_trans_hat * np.cos(p.theta + delta_rot1_hat)
                    p.y += delta_trans_hat * np.sin(p.theta + delta_rot1_hat)
                    p.theta += delta_rot1_hat + delta_rot2_hat
                    p.theta = self.angle_diff(p.theta, 0)  # Normalizar el ángulo
            else:
                for p in self.particles:
                    noise_theta = np.random.normal(0, alpha1 * abs(delta_theta))
                    p.theta += delta_theta + noise_theta
                    p.theta = self.angle_diff(p.theta, 0)

        # OK TODO: 2. Measurement update (weight particles)
        weights = []
        for p in self.particles:
            weight = self.compute_weight(p, msg) # Compute weights for each particle
            weights.append(weight)
            #p.weight = weight # Esto es el save??

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            # Si todos los pesos son cero, asigna pesos uniformes
            weights = [1.0 / self.num_particles] * self.num_particles

        for i, p in enumerate(self.particles):
            p.weight = weights[i] # Resave weights

        # 3. Resample
        self.particles = self.resample_particles(self.particles)

        # OK TODO: 4. Use weighted mean of all particles for mapping and pose (update current_map_pose and current_odom_pose, for each particle)
        mean_x = sum(p.x * p.weight for p in self.particles)
        mean_y = sum(p.y * p.weight for p in self.particles)

        # Para el ángulo:
        mean_sin = sum(np.sin(p.theta) * p.weight for p in self.particles)
        mean_cos = sum(np.cos(p.theta) * p.weight for p in self.particles)
        mean_theta = math.atan2(mean_sin, mean_cos)

        # Guardamos la pose estimada
        self.current_map_pose = [mean_x, mean_y, mean_theta]
        # self.current_odom_pose = self.last_odom.pose.pose.position  # Odom pose es el ultimo odom message 
        self.current_odom_pose = odom_pose
        # 5. Mapping (update map with best particle's pose)
        for p in self.particles:
            self.update_map(p, msg)

        # 6. Broadcast map->odom transform
        self.broadcast_map_to_odom()

    def compute_weight(self, particle, scan_msg):
        # Simple likelihood: count how many endpoints match occupied cells
        score = 0.0
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        val_readings = 0

        for i, range_dist in enumerate(scan_msg.ranges):
            # OK TODO: Filter readings based on lidar distance limits
            if (range_dist < scan_msg.range_min or range_dist > scan_msg.range_max or 
                math.isnan(range_dist) or 
                range_dist < self.lidar_min_distance or 
                range_dist > self.lidar_max_distance):
                continue

            # TODO: Compute the map coordinates of the endpoint: transform the scan into the map frame
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            
            end_x = robot_x + range_dist * math.cos(robot_theta + angle)
            end_y = robot_y + range_dist * math.sin(robot_theta + angle)
            
            cell_x = int((end_x - self.map_origin_x) / self.resolution)
            cell_y = int((end_y - self.map_origin_y) / self.resolution)

            # TODO: Use particle.log_odds_map for scoring
            if 0 <= cell_x < self.map_width_cells and 0 <= cell_y < self.map_height_cells:
                if particle.log_odds_map[cell_y, cell_x] > 0:
                    score += 1.0
                elif particle.log_odds_map[cell_y, cell_x] < -1.0:
                    score -= 1.0
                val_readings += 1
        
        if val_readings > 0:
            return max(score / val_readings + 1.0, 0.1)
        else:
            return 0.1


    def resample_particles(self, particles):
        # TODO: Resample particles
        new_particles = []
        weights = [p.weight for p in particles]
        
        # Verificar si hay diversidad suficiente - más agresivo para 10 partículas
        max_weight = max(weights)
        effective_particles = sum(1 for w in weights if w > max_weight * 0.05)  # Umbral más bajo
        
        # Si hay poca diversidad, añadir partículas aleatorias - más frecuente para 10 partículas
        if effective_particles < len(particles) * 0.5:  # Umbral más alto
            self.add_random_particles(particles, 0.4)  # 40% partículas aleatorias para 10 partículas
        
        # Systematic resampling
        total_weight = sum(weights)
        if total_weight <= 0:
            # Si todos los pesos son cero, reinicializar
            return [Particle(0.0, 0.0, 0.0, 1.0/len(particles), 
                           (self.map_height_cells, self.map_width_cells)) 
                   for _ in range(len(particles))]
        
        weights = [w / total_weight for w in weights]
        
        r = np.random.uniform(0, 1.0/len(particles))
        c = weights[0]
        i = 0
        for m in range(len(particles)):
            u = r + m / len(particles)
            while u > c and i < len(weights) - 1:
                i += 1
                c += weights[i]
            
            if i < len(particles):
                # Create a copy of the particle
                p = particles[i]
                new_particle = Particle(p.x, p.y, p.theta, 1.0/len(particles), 
                                      (self.map_height_cells, self.map_width_cells))
                new_particle.log_odds_map = p.log_odds_map.copy()
                new_particles.append(new_particle)
        
        return new_particles if len(new_particles) == len(particles) else particles

    def add_random_particles(self, particles, fraction):
        """Añade partículas aleatorias para mantener diversidad"""
        num_random = int(len(particles) * fraction)
        best_particle = max(particles, key=lambda p: p.weight)
        
        for i in range(num_random):
            if i < len(particles):
                # Partícula aleatoria cerca de la mejor - más ruido para 10 partículas
                noise_x = np.random.normal(0, 0.5)  # Más ruido en X
                noise_y = np.random.normal(0, 0.5)  # Más ruido en Y
                noise_theta = np.random.normal(0, 0.3)  # Más ruido en theta
                
                particles[i].x = best_particle.x + noise_x
                particles[i].y = best_particle.y + noise_y
                particles[i].theta = self.angle_diff(best_particle.theta + noise_theta, 0)
                particles[i].weight = 1.0 / len(particles)
                particles[i].log_odds_map = best_particle.log_odds_map.copy()

    def update_map(self, particle, scan_msg):

        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta

        for i, range_dist in enumerate(scan_msg.ranges):
            is_hit = range_dist < scan_msg.range_max
            current_range = min(range_dist, scan_msg.range_max)
            # OK TODO: Filter readings based on lidar distance limits
            if (math.isnan(current_range) or 
                current_range < scan_msg.range_min or
                current_range < self.lidar_min_distance or 
                current_range > self.lidar_max_distance):
                continue
            # OK TODO: Update map: transform the scan into the map frame
            # OK TODO: Use self.bresenham_line for free cells
            # OK TODO: Update particle.log_odds_map accordingly

            # --- 1. Transformar el rayo al marco del mapa
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            angle_world = robot_theta + angle

            hit_x = robot_x + current_range * math.cos(angle_world)
            hit_y = robot_y + current_range * math.sin(angle_world)

            # Convertir a coordenadas del mapa (índices de celda)
            map_x0 = int((robot_x - self.map_origin_x) / self.resolution) # posicion del robot
            map_y0 = int((robot_y - self.map_origin_y) / self.resolution) # posicion del robot
            map_x1 = int((hit_x - self.map_origin_x) / self.resolution) # posicion del hit
            map_y1 = int((hit_y - self.map_origin_y) / self.resolution) # posicion del hit

            # --- 2. Raytrace celdas libres
            self.bresenham_line(particle, map_x0, map_y0, map_x1, map_y1)

            # --- 3. Actualizar celda de impacto como ocupada (si es hit)
            if 0 <= map_x1 < self.map_width_cells and 0 <= map_y1 < self.map_height_cells:
                if is_hit:
                    particle.log_odds_map[map_y1, map_x1] += self.log_odds_occupied
                    particle.log_odds_map[map_y1, map_x1] = np.clip(particle.log_odds_map[map_y1, map_x1], self.log_odds_min, self.log_odds_max)
        
            
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
        # Use best particle instead of average for better map consistency
        # Find the particle with the highest weight
        best_particle = max(self.particles, key=lambda p: p.weight)
        
        # Use the best particle's map
        best_log_odds_map = best_particle.log_odds_map

        # Crear mensaje OccupancyGrid
        map_msg = OccupancyGrid()

        # Header
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value

        # Info del mapa
        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.map_width_cells
        map_msg.info.height = self.map_height_cells
        map_msg.info.origin.position.x = self.map_origin_x
        map_msg.info.origin.position.y = self.map_origin_y
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0  # sin rotación

        # Convertir log-odds del mejor particle a formato de ROS (0 = libre, 100 = ocupado, -1 = desconocido)
        occupancy_grid = np.zeros((self.map_height_cells, self.map_width_cells), dtype=np.int8)
        for y in range(self.map_height_cells):
            for x in range(self.map_width_cells):
                log_odds = best_log_odds_map[y, x]
                if log_odds > 0:
                    occupancy_grid[y, x] = 100  # Occupied
                elif log_odds < 0:
                    occupancy_grid[y, x] = 0    # Free
                else:
                    occupancy_grid[y, x] = -1   # Unknown

        map_msg.data = occupancy_grid.flatten().tolist()
        self.map_publisher.publish(map_msg)
        self.get_logger().debug("Map published.")

    def broadcast_map_to_odom(self):
        
        # OK TODO: Broadcast map->odom transform
        if not hasattr(self, 'current_map_pose'):
            return

        x, y, theta = self.current_map_pose

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value
        t.child_frame_id = self.get_parameter('odom_frame').get_parameter_value().string_value

        # The transform from map to odom is the difference between the robot's estimated pose in the map
        # and its pose from odometry. However, this subtraction must be done correctly,
        # taking the rotation between frames into account.
        map_to_odom_x = self.current_map_pose[0] - self.current_odom_pose[0]
        map_to_odom_y = self.current_map_pose[1] - self.current_odom_pose[1]
        map_to_odom_theta = self.angle_diff(self.current_map_pose[2], self.current_odom_pose[2])
        
        t.transform.translation.x = map_to_odom_x
        t.transform.translation.y = map_to_odom_y
        t.transform.translation.z = 0.0

        q = euler2quat(0, 0, map_to_odom_theta)
        t.transform.rotation.w = q[0]
        t.transform.rotation.x = q[1]
        t.transform.rotation.y = q[2]
        t.transform.rotation.z = q[3]

        self.tf_broadcaster.sendTransform(t)

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