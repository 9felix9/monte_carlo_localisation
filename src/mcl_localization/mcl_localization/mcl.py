import csv
import numpy as np
from landmark_manager import LandmarkManager

class MCL: 

    def __init__(self):
        
        self.particle_map = {}
        self.landmarks = {}

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
        
        # put coordinates in matrix and just stack the columns in a Particle Matrix
        self.Particles = np.column_stack((xs, ys, thetas, weights))
        


if __name__ == "__main__":
    mcl = MCL()
    mcl.initializeParticles()        