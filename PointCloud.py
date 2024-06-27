import numpy as np
import laspy


class PointCloud:
    def __init__(self, points: np.ndarray):
        self.points = points

        def crop_to_aabb(self, min_bound: np.ndarray, max_bound: np.ndarray) -> 'PointCloud':
            """
            Crop the point cloud to an axis-aligned bounding box.

            :param min_bound: numpy array of shape (3,) representing the minimum x, y, z coordinates of the bounding box
            :param max_bound: numpy array of shape (3,) representing the maximum x, y, z coordinates of the bounding box
            :return: A new PointCloud object containing only the points within the bounding box
            """
            # Create a boolean mask for points within the bounding box
            mask = np.all((self.points >= min_bound) & (self.points <= max_bound), axis=1)
            cropped_points = self.points[mask]
            return PointCloud(cropped_points)

        def write_to_las(self, filename: str):
            """
            Write the point cloud to a LAS file.

            :param filename: The name of the output LAS file
            """
            # Create a new LAS file
            las = laspy.create(file_version="1.2", point_format=2)

            # Set the header
            las.header.offsets = np.min(self.points, axis=0)
            las.header.scales = np.array([0.001, 0.001, 0.001])

            # Set x, y, z
            las.x = self.points[:, 0]
            las.y = self.points[:, 1]
            las.z = self.points[:, 2]

            # Write the LAS file
            las.write(filename)
            print(f"Point cloud written to {filename}")
