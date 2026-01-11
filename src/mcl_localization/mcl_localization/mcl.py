import math
import numpy as np
from landmark_manager import LandmarkManager

class MCL: 

    def __init__(self):
        
        # those variables will be set by the tests at the end 
        self.sigma_trans = 0.01
        self.sigma_rotation = 0.01

        # particle matrix in shape N,3 (x, y, theta)
        self.Particles = None
        
        self.particle_map = {}
        self.landmarks = {} # maybe dont need lets see

    def initializeParticles(self): 
        number_of_particles = 100 # could be something else dont know   

        lm = LandmarkManager()
        lm.load_from_csv("/home/felix/Schreibtisch/projects/robotics_hw2/src/mcl_localization/landmarks.csv") # LMs saved in lm obj

        # get min and max from x and y
        landmarks = lm.get_all_landmarks()
        coordinates = [coord for coord in landmarks.values()]
        lm_coordinates = np.array(coordinates)

        x_min, y_min = lm_coordinates.min(axis=0)
        x_max, y_max = lm_coordinates.max(axis=0)

        # sample points from the found range  
        xs = np.random.uniform(x_min, x_max, size=number_of_particles)
        ys = np.random.uniform(y_min, y_max, size=number_of_particles)
        thetas = np.random.uniform(-np.pi, np.pi, size=number_of_particles)
        weights = np.ones(number_of_particles) / number_of_particles
        
        # put coordinates in matrix and just stack the columns in a Particle Matrix and save it
        self.Particles = np.column_stack((xs, ys, thetas))
        self.particle_weights = np.copy(weights)
        
        # return all Particles for eventually future use
        return self.Particles, self.particle_weights
    
    # Propagate all particles using a velocity motion model with Gaussian noise
    # in the lecture a odometry model is presented - maybe use that instead
    def motionUpdate(self, vx, vy, vtheta, dt):

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