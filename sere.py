
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import tf_transformations
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from scipy.ndimage import gaussian_filter

class Particle:
    def __init__(self, x, y, theta, weight, map_shape):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
        self.log_odds_map = np.zeros(map_shape, dtype=np.float64)

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
        self.declare_parameter('map_width_meters', 5.0)
        self.declare_parameter('map_height_meters', 5.0)
        self.declare_parameter('num_particles', 10)
        self.declare_parameter('max_lidar_range', 3.0)
        self.declare_parameter('num_beams', 90)
        self.declare_parameter('smoothing_sigma', 0.5)

        self.resolution = self.get_parameter('map_resolution').get_parameter_value().double_value
        self.map_width_m = self.get_parameter('map_width_meters').get_parameter_value().double_value
        self.map_height_m = self.get_parameter('map_height_meters').get_parameter_value().double_value
        self.num_beams = self.get_parameter('num_beams').get_parameter_value().integer_value
        self.smoothing_sigma = self.get_parameter('smoothing_sigma').get_parameter_value().double_value
        self.map_width_cells = int(self.map_width_m / self.resolution)
        self.map_height_cells = int(self.map_height_m / self.resolution)
        self.map_origin_x = -self.map_width_m / 2.0
        self.map_origin_y = -5.0

        self.max_lidar_range = self.get_parameter('max_lidar_range').get_parameter_value().double_value

        # TODO: define the log-odds criteria for free and occupied cells
        self.log_odds_occ = 2.5
        self.log_odds_free = -0.2

        self.lock_occ = 4.0
        self.lock_free = -4.0

        self.log_odds_max = 5.0
        self.log_odds_min = -5.0

        

        # Particle filter
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.particles = [
            Particle(0.0, 0.0, 0.0, 1.0/self.num_particles,
                     (self.map_height_cells, self.map_width_cells))
            for _ in range(self.num_particles)
        ]
        self.last_odom = None

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
            rclpy.qos.qos_profile_sensor_data
        )

        self.get_logger().info("Python SLAM node with particle filter initialized.")
        self.map_publish_timer = self.create_timer(1.0, self.publish_map)

    def odom_callback(self, msg: Odometry):
        # Store odometry for motion update
        self.last_odom = msg

    def scan_callback(self, msg: LaserScan):
        if self.last_odom is None:
            return

        # --- FILTRADO DE REBOTES: convierte rangos fuera de confianza en NaN ---
        filtered = []
        for r in msg.ranges:
            if math.isnan(r) or r < msg.range_min or r >= self.max_lidar_range - 0.05:
                filtered.append(float('nan'))
            else:
                filtered.append(r)
        # Sobrescribimos msg.ranges para que compute_weight() y update_map() los ignoren
        msg.ranges = filtered

        # 1. Motion update (sample motion model)
        odom = self.last_odom
        # TODO: Retrieve odom_pose from odom message - remember that orientation is a quaternion
        odom_x = odom.pose.pose.position.x
        odom_y = odom.pose.pose.position.y
        quat = odom.pose.pose.orientation
        _, _, odom_theta = tf_transformations.euler_from_quaternion(
            [quat.x, quat.y, quat.z, quat.w])

        # TODO: Model the particles around the current pose
        for p in self.particles:
            # Add noise to simulate motion uncertainty
            p.x = odom_x + np.random.normal(0, 0.02)
            p.y = odom_y + np.random.normal(0, 0.02)
            p.theta = odom_theta + np.random.normal(0, 0.01)

        # TODO: 2. Measurement update (weight particles)
        weights = []
        # Parámetros del beam model
        z_hit, z_rand, sigma_hit = 0.8, 0.2, 0.1
        beam_idxs = np.linspace(0, len(msg.ranges)-1, self.num_beams, dtype=int)

        for p in self.particles:
            w = 1.0
            for i in beam_idxs:
                z = msg.ranges[i]
                if math.isnan(z) or z < msg.range_min or z > self.max_lidar_range:
                    continue

                angle = msg.angle_min + i * msg.angle_increment

                # Ray-casting en el mapa de la partícula para rango esperado
                expected = self.max_lidar_range
                r_test = 0.0
                while r_test < self.max_lidar_range:
                    x_test = p.x + r_test * math.cos(p.theta + angle)
                    y_test = p.y + r_test * math.sin(p.theta + angle)
                    cx = int((x_test - self.map_origin_x) / self.resolution)
                    cy = int((y_test - self.map_origin_y) / self.resolution)
                    if not (0 <= cx < self.map_width_cells and 0 <= cy < self.map_height_cells):
                        break
                    if p.log_odds_map[cy, cx] > 0:
                        expected = r_test
                        break
                    r_test += self.resolution

                # p_hit: componente gaussiana
                p_hit = (1.0 / (math.sqrt(2*math.pi) * sigma_hit)) * \
                        math.exp(-0.5 * ((z - expected) / sigma_hit)**2)
                # p_rand: componente uniforme de ruido
                p_rand = z_rand / self.max_lidar_range
                # mezcla de probabilidades
                beam_p = z_hit * p_hit + p_rand

                w *= beam_p

            weights.append(w + 1e-9)

        # Normalize weights
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]
        else:
            weights = [1.0 / self.num_particles for _ in weights]

        for i, p in enumerate(self.particles):
            p.weight = weights[i]

        # 3. Resample
        self.particles = self.resample_particles(self.particles)

        # TODO: 4. Use weighted mean of all particles for mapping and pose (update current_map_pose and current_odom_pose, for each particle)
        mean_x = sum(p.x * p.weight for p in self.particles)
        mean_y = sum(p.y * p.weight for p in self.particles)
        mean_sin = sum(math.sin(p.theta) * p.weight for p in self.particles)
        mean_cos = sum(math.cos(p.theta) * p.weight for p in self.particles)
        mean_theta = math.atan2(mean_sin, mean_cos)
        self.current_map_pose = (mean_x, mean_y, mean_theta)

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
            if range_dist < scan_msg.range_min or range_dist > self.max_lidar_range or math.isnan(range_dist):
                continue

            if range_dist > 0.9 * self.max_lidar_range:
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
        return score + 1e-6

    def resample_particles(self, particles):
        # TODO: Resample particles
        new_particles = []
        weights = [p.weight for p in particles]
        cum_weights = np.cumsum(weights)
        for _ in particles:
            r = np.random.rand()
            idx = np.searchsorted(cum_weights, r)
            selected = particles[idx]
            new_p = Particle(
                selected.x, selected.y, selected.theta,
                1.0/self.num_particles,
                (self.map_height_cells, self.map_width_cells)
            )
            new_p.log_odds_map = selected.log_odds_map.copy()
            new_particles.append(new_p)
        return new_particles

    def update_map(self, particle, scan_msg):
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        for i, range_dist in enumerate(scan_msg.ranges):

            if math.isnan(range_dist) or range_dist < scan_msg.range_min or range_dist >= self.max_lidar_range - 0.05:
                continue

            if range_dist > 0.9 * self.max_lidar_range:
            # drop corner‐bounces and max‐range returns
                continue

            is_hit = range_dist < self.max_lidar_range
            current_range = min(range_dist, self.max_lidar_range)
            if math.isnan(current_range) or current_range < scan_msg.range_min:
                continue
            # TODO: Update map: transform the scan into the map frame
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            end_x = robot_x + current_range * math.cos(robot_theta + angle)
            end_y = robot_y + current_range * math.sin(robot_theta + angle)
            cell_x = int((end_x - self.map_origin_x) / self.resolution)
            cell_y = int((end_y - self.map_origin_y) / self.resolution)
            # TODO: Use self.bresenham_line for free cells
            start_x = int((robot_x - self.map_origin_x) / self.resolution)
            start_y = int((robot_y - self.map_origin_y) / self.resolution)
            self.bresenham_line(particle, start_x, start_y, cell_x, cell_y)
            # TODO: Update particle.log_odds_map accordingly
            if 0 <= cell_x < self.map_width_cells and 0 <= cell_y < self.map_height_cells:

                current = particle.log_odds_map[cell_y, cell_x]
                
                if is_hit:

                    if current > self.lock_free:
                        particle.log_odds_map[cell_y, cell_x] = min(current + self.log_odds_occ, self.log_odds_max)
                    # particle.log_odds_map[cell_y, cell_x] += self.log_odds_occ
                else:

                    if current < self.lock_occ:
                        particle.log_odds_map[cell_y, cell_x] = max(current + self.log_odds_free, self.log_odds_min)
                    # particle.log_odds_map[cell_y, cell_x] += self.log_odds_free
                particle.log_odds_map[cell_y, cell_x] = np.clip(
                    particle.log_odds_map[cell_y, cell_x],
                    self.log_odds_min,
                    self.log_odds_max
                )

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
                particle.log_odds_map[y0, x0] = np.clip(
                    particle.log_odds_map[y0, x0],
                    self.log_odds_min,
                    self.log_odds_max
                )
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

        # Combine particle maps into one log-odds grid
        combined_log_odds = np.zeros((self.map_height_cells, self.map_width_cells), dtype=np.float32)
        for p in self.particles:
            combined_log_odds += p.log_odds_map * p.weight

        combined_log_odds = gaussian_filter(combined_log_odds, sigma=self.smoothing_sigma)

        # Convert log-odds to probability

        prob_map = 1.0 / (1.0 + np.exp(-combined_log_odds))
        data = (prob_map * 100).astype(np.int8)
        map_msg.data = data.flatten().tolist()

        self.map_publisher.publish(map_msg)
        self.get_logger().debug("Map published.")
    

    def broadcast_map_to_odom(self):
        # TODO: Broadcast map->odom transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value
        t.child_frame_id = self.get_parameter('odom_frame').get_parameter_value().string_value
        t.transform.translation.x = self.current_map_pose[0]
        t.transform.translation.y = self.current_map_pose[1]
        t.transform.translation.z = 0.0
        quat = tf_transformations.quaternion_from_euler(0, 0, self.current_map_pose[2])
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
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