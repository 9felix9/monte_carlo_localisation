import csv

class LandmarkManager:
    def __init__(self):
        self.landmarks = {}

    def load_from_csv(self, csv_path):
        """
        Load landmarks from CSV file

        STUDENT TODO:
        1. Open the CSV file at csv_path
        2. Parse each line as: id,x,y
        3. Handle comments (lines starting with #)
        4. Store landmark positions in the landmarks_ map
        5. Return true if successful, false otherwise
        """
        try:
            with open(csv_path, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row[0].startswith('#') or len(row) < 3:
                        continue
                    landmark_id = int(row[0])
                    x = float(row[1])
                    y = float(row[2])
                    self.landmarks[landmark_id] = (x, y)
            return True
        except Exception as e:
            print(f'Error loading landmarks: {e}')
            return False

    def get_landmark(self, landmark_id):
        """
        Get landmark position by ID
        """
        return self.landmarks.get(landmark_id, (None, None))

    def get_all_landmarks(self):
        """
        Get all landmarks
        """
        return self.landmarks

    def get_landmarks_in_radius(self, x, y, radius):
        """
        Get landmarks within a certain radius of a point

        STUDENT TODO:
        1. Iterate through all landmarks
        2. Calculate distance from (x, y) to each landmark
        3. Add landmarks within radius to result list
        4. Return the list of landmark IDs
        """
        result = []
        for landmark_id, (lx, ly) in self.landmarks.items():
            if self.distance(x, y, lx, ly) <= radius:
                result.append(landmark_id)
        return result

    @staticmethod
    def distance(x1, y1, x2, y2):
        """
        Calculate Euclidean distance between two points
        """
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5