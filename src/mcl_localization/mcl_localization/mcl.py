import math
import numpy as np

class MCL:

    def __init__(self):
        
        # those variables will be set by the tests at the end 
        self.sigma_trans = 0.01
        self.sigma_rotation = 0.01
        self.sigma_sensor_noise = 0.01
        self.sample_threshold = 0.5
        
        # particle matrix in shape N,3 (x, y, theta)
        self.number_of_particles = 100 # could be something else dont know   
        self.Particles = None
        
        self.particle_map = {}
        self.landmarks_gt = {} # maybe dont need lets see
        self.landmarks_observed = {}

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
    def motionUpdate(self, vx, vy, vtheta, dt):
        '''
        Implements the motionModel which constructs "new" particles according
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
        dx_noise = dx - noise_x
        dy_noise = dy - noise_y
        dtheta_noise = dtheta - noise_theta

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
           # Todo: implement likelihood for the particles with the landmarks_observed
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
                u = r + m / n
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

if __name__ == "__main__":
    mcl = MCL()
    mcl.initializeParticles()        