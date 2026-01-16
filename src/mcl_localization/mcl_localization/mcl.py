import math
import numpy as np

class MCL:

    def __init__(self, 
                 logger, 
                 number_of_particles = 100,
                 sigma_sensor_noise = 0.5,
                 alpha1 = 0.05, 
                 alpha2 = 0.05,
                 alpha3 = 0.10, 
                 alpha4 = 0.05,
                 ):

        # Odometry model noise parameters 
        self.alpha1 = alpha1 # rot noise from rot
        self.alpha2 = alpha2 # rot noise from trans
        self.alpha3 = alpha3  # trans noise from trans
        self.alpha4 = alpha4  # trans noise from rot

        self.sigma_sensor_noise = sigma_sensor_noise

        # variables for velocity modtion model 
        # self.sigma_trans = 0.2
        # self.sigma_rotation = 0.2
        # self.sample_threshold = 0.5 # for N_eff

        
        # particle matrix in shape N,3 (x, y, theta)
        self.number_of_particles = number_of_particles # could be something else dont know   
        self.Particles = None
        self.particle_weights = None

        self.particle_map = {}
        self.landmarks_gt = {} # maybe dont need lets see
        self.landmarks_observed = {}

        self.logger = logger
        # np.set_printoptions(
        #     precision=3,      # Nachkommastellen
        #     suppress=True,    # Keine wissenschaftliche Notation mit e
        #     linewidth=120     # Zeilenbreite
        #     )

    def initializeParticles(self): 

        coordinates = [coord for coord in self.landmarks_gt.values()]
        lm_coordinates = np.array(coordinates)

        x_min, y_min = lm_coordinates.min(axis=0)
        x_max, y_max = lm_coordinates.max(axis=0)

        # sample points from the found range  
        xs = np.random.uniform(x_min, x_max, size=self.number_of_particles)
        ys = np.random.uniform(y_min, y_max, size=self.number_of_particles)
        thetas = np.random.uniform(-np.pi, np.pi, size=self.number_of_particles)
        weights = np.ones(self.number_of_particles) / self.number_of_particles
        
        # put coordinates in matrix and just stack the columns in a Particle Matrix and save it
        self.Particles = np.column_stack((xs, ys, thetas))
        self.particle_weights = np.copy(weights)
        
        # return all Particles for eventually future use
        return self.Particles, self.particle_weights
    
    # Propagate all particles using a velocity motion model with Gaussian noise
    # in the lecture a odometry model is presented - maybe use that instead
    def motionUpdateVelocity(self, vx, vy, vtheta, dt):
        '''
        Implements the velocity motionModel which constructs "new" particles according
        to the ones that are already in the self.Particles Matrix (out of the resampling method from prev iteration)
        
        :param vx: x velocity
        :param vy: y velocity
        :param vtheta: angle velocity
        :param dt: time delta to prev state

        :return Particle Matrix (N,3) 
        '''

        # Number of particles
        n = self.Particles.shape[0]

        # ------------------------------------------------------------------
        # 1) Mean motion in the robot frame (control input)
        #    These are the expected displacements given the odometry
        # ------------------------------------------------------------------
        dx = vx * dt
        dy = vy * dt
        dtheta = vtheta * dt

        # ------------------------------------------------------------------
        # 2) Sample Gaussian noise for each particle
        #    This represents uncertainty in the robot motion
        # ------------------------------------------------------------------
        noise_x = np.random.normal(0.0, self.sigma_trans, n)
        noise_y = np.random.normal(0.0, self.sigma_trans, n)
        noise_theta = np.random.normal(0.0, self.sigma_rotation, n)

        # Noisy motion in the robot frame (one sample per particle)
        dx_noise = dx + noise_x
        dy_noise = dy + noise_y
        dtheta_noise = dtheta + noise_theta

        # ------------------------------------------------------------------
        # 3) Transform noisy motion from robot frame to world frame
        #    Each particle has its own orientation theta_i
        # ------------------------------------------------------------------
        theta = self.Particles[:, 2]

        delta_x = np.cos(theta) * dx_noise - np.sin(theta) * dy_noise
        delta_y = np.sin(theta) * dx_noise + np.cos(theta) * dy_noise

        # ------------------------------------------------------------------
        # 4) Apply the motion update to all particles
        #    This is: x_i ← x_i + Δx_i, y_i ← y_i + Δy_i, θ_i ← θ_i + Δθ_i
        # ------------------------------------------------------------------
        self.Particles[:, 0] += delta_x
        self.Particles[:, 1] += delta_y
        self.Particles[:, 2] += dtheta_noise

        # ------------------------------------------------------------------
        # 5) Normalize angles to the interval [-pi, pi]
        # ------------------------------------------------------------------
        normalize_angle_vectorized = np.vectorize(self.normalize_angle)
        self.Particles[:, 2] = normalize_angle_vectorized(self.Particles[:, 2])

        return self.Particles
    
    def motionUpdateOdometry(self, delta_rot1, delta_trans, delta_rot2):
        """
        Odometry-based motion model
        Inputs are measured odometry increments:
        delta_rot1, delta_trans, delta_rot2
        Updates all particles in-place.
        """
        n = self.Particles.shape[0]

        # Variances as in the lecture model:
        # p1: rot1 noise variance = alpha1*rot1^2 + alpha2*trans^2
        # p2: trans noise variance = alpha3*trans^2 + alpha4*rot1^2 + alpha4*rot2^2
        # p3: rot2 noise variance = alpha1*rot2^2 + alpha2*trans^2
        var_rot1 = self.alpha1 * (delta_rot1 ** 2) + self.alpha2 * (delta_trans ** 2)
        var_trans = self.alpha3 * (delta_trans ** 2) + self.alpha4 * (delta_rot1 ** 2) + self.alpha4 * (delta_rot2 ** 2)
        var_rot2 = self.alpha1 * (delta_rot2 ** 2) + self.alpha2 * (delta_trans ** 2)

        # Convert to std dev; guard against 0
        std_rot1 = np.sqrt(max(var_rot1, 1e-12))
        std_trans = np.sqrt(max(var_trans, 1e-12))
        std_rot2 = np.sqrt(max(var_rot2, 1e-12))

        # Sample noisy increments per particle
        delta_rot1_hat = delta_rot1 + np.random.normal(0.0, std_rot1, n)
        delta_trans_hat = delta_trans + np.random.normal(0.0, std_trans, n)
        delta_rot2_hat = delta_rot2 + np.random.normal(0.0, std_rot2, n)

        theta = self.Particles[:, 2]

        # Apply motion (standard odom model update)
        self.Particles[:, 0] += delta_trans_hat * np.cos(theta + delta_rot1_hat)
        self.Particles[:, 1] += delta_trans_hat * np.sin(theta + delta_rot1_hat)
        self.Particles[:, 2] += delta_rot1_hat + delta_rot2_hat

        # Normalize to [-pi, pi] (fast vectorized)
        self.Particles[:, 2] = (self.Particles[:, 2] + np.pi) % (2.0 * np.pi) - np.pi

        return self.Particles

    
    def measurementUpdate(self): 
        n = self.Particles.shape[0]
        eps = 1e-12 # just a very small number not not get 0 somewhere
        particles_x_vec = self.Particles[:,0]
        particles_y_vec = self.Particles[:,1]
        particles_theta_vec = self.Particles[:,2] 

        cos_theta_vec = np.cos(particles_theta_vec)
        sin_theta_vec = np.sin(particles_theta_vec)

        predicted_in_robot = {} 

        # for each obs landmark find the gt landmark with the id
        # for each particle check the distance between the gt landmark and the particle
        # transfer this into robot frame and save the new x-y vectors for all particles
    

        new_particle_weights = np.ones(n)
        for lm_id, (x_obs, y_obs) in self.landmarks_observed.items():
           
           x_map, y_map = self.landmarks_gt[lm_id] 

           dx = x_map - particles_x_vec # diff between particle and map coord
           dy = y_map - particles_y_vec
           
           # transform world --> robot frame
           x_robot = cos_theta_vec * dx + sin_theta_vec * dy 
           y_robot = -sin_theta_vec * dx + cos_theta_vec * dy
           
           # is numpy conform - x_robot and y_robot both have the length (rows) of the Particle Matrix
           predicted_in_robot[lm_id] = (x_robot, y_robot)

           #likelihood 
           likelihood_per_lm = np.exp(-((x_obs - x_robot)**2 + (y_obs - y_robot)**2) / (2.0*self.sigma_sensor_noise**2))
           # question for likelihood: If robots at pose x, how likely it would be to see oberservation z
           new_particle_weights *= likelihood_per_lm

        normalize_factor = np.sum(new_particle_weights) 

        if normalize_factor < eps: 
            new_particle_weights[:] = 1.0 / n

        else: 
            new_particle_weights /= normalize_factor
        

        self.particle_weights = new_particle_weights

        
        return self.particle_weights
    

    def resampling(self): 
        '''
        Implements the low-variance resampling process where only the best fitting
        particles come into the new particles matrix according to their weights.
        
        '''
        n = self.Particles.shape[0]

        # normalize weights for safety - should be done in prev step
        self.particle_weights /= np.sum(self.particle_weights)

        number_of_eff_particles = 1 / np.sum(self.particle_weights**2)
        number_of_eff_particles = number_of_eff_particles
        

        if number_of_eff_particles < self.sample_threshold*n: 
            # here ist low variance resample algo
            Resample_particles = np.zeros_like(self.Particles)
            r = np.random.uniform(0.0, 1.0/n)
            w = self.particle_weights[0]
            i = 0

            for m in range(n):
                u = r + float(m) / float(n)
                while u > w:
                    i += 1
                    w += self.particle_weights[i]
                
                Resample_particles[m] = self.Particles[i]
        
            # after the resampling process set equal weights
        
            self.Particles = Resample_particles
            self.particle_weights = np.ones(n) / n


    def estimatePose(self): 

        weighted_mean = np.sum(self.Particles * self.particle_weights[:, None], axis=0)

        theta = self.Particles[:,2]
        sin_mean = np.sum(np.sin(theta) * self.particle_weights)
        cos_mean = np.sum(np.cos(theta) * self.particle_weights)
        weighted_mean[2] = np.arctan2(sin_mean, cos_mean) 

        return weighted_mean

    # ======================================================================
    # Angle Normalization
    # ======================================================================
    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    

    def log_particles(self, stage: str, k_top=5, k_rand=5):
        P = self.Particles
        w = self.particle_weights

        if P is None or w is None:
            self.logger.warn(f"[{stage}] No particles or weights")
            return

        n = len(w)
        self.logger.info(f"\n[{stage}] N={n}")

        # sort by weight descending
        idx_sorted = np.argsort(w)[::-1]

        self.logger.info("Top particles by weight:")
        for i in range(min(k_top, n)):
            idx = idx_sorted[i]
            x, y, th = P[idx]
            self.logger.info(
                f"  {i+1:02d}) idx={idx:4d}  w={w[idx]:.6f}  p=({x:.3f},{y:.3f},{th:.3f})"
            )

        self.logger.info("Random particles:")
        rnd_idx = np.random.choice(n, size=min(k_rand, n), replace=False)
        for idx in rnd_idx:
            x, y, th = P[idx]
            self.logger.info(
                f"  idx={idx:4d}  w={w[idx]:.6f}  p=({x:.3f},{y:.3f},{th:.3f})"
            )

