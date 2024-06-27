import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d

from PointCloud import PointCloud


def estimate_normals(pcd, radius=0.1, max_nn=30):
    """Estimate normals for a point cloud."""
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd.points)
    pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return np.asarray(pcd_o3d.normals)


def point_to_plane_icp(source, target, max_iterations=50, tolerance=1e-6):
    """Perform point-to-plane ICP alignment."""
    source_points = np.asarray(source.points, dtype=np.float64)
    target_points = np.asarray(target.points, dtype=np.float64)

    # Estimate normals for the target point cloud
    target_normals = estimate_normals(target)

    # Create a KD-tree for efficient nearest neighbor search
    target_tree = cKDTree(target_points)

    # Initial transformation
    transformation = np.eye(4, dtype=np.float64)

    for iteration in range(max_iterations):
        # Find nearest neighbors
        distances, indices = target_tree.query(source_points, k=1)

        # Get corresponding points and normals
        corresponding_points = target_points[indices]
        corresponding_normals = target_normals[indices]

        # Compute point-to-plane distances
        vectors = source_points - corresponding_points
        point_to_plane_distances = np.abs(np.sum(vectors * corresponding_normals, axis=1))

        # Check for convergence
        if np.mean(point_to_plane_distances) < tolerance:
            break

        # Construct the linear system for point-to-plane minimization
        A = np.zeros((len(source_points), 6), dtype=np.float64)
        b = np.zeros(len(source_points), dtype=np.float64)

        for i, (p, n) in enumerate(zip(source_points, corresponding_normals)):
            A[i] = [n[2] * p[1] - n[1] * p[2],
                    n[0] * p[2] - n[2] * p[0],
                    n[1] * p[0] - n[0] * p[1],
                    n[0], n[1], n[2]]
            b[i] = np.dot(n, corresponding_points[i] - p)

        # Solve the linear system
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # Convert solution to transformation matrix
        delta_rotation = np.array([
            [0, -x[2], x[1]],
            [x[2], 0, -x[0]],
            [-x[1], x[0], 0]
        ], dtype=np.float64)
        delta_rotation = np.eye(3, dtype=np.float64) + delta_rotation
        delta_translation = x[3:6]

        delta_transform = np.eye(4, dtype=np.float64)
        delta_transform[:3, :3] = delta_rotation
        delta_transform[:3, 3] = delta_translation

        # Update the transformation
        transformation = np.dot(delta_transform, transformation)

        # Apply the transformation to the source points
        source_points = np.dot(delta_rotation, source_points.T).T + delta_translation

    return transformation


def fine_align_point_clouds(source: PointCloud, target: PointCloud) -> PointCloud:
    """Perform fine alignment of source point cloud to target point cloud."""
    transformation = point_to_plane_icp(source, target)

    # Apply the final transformation
    aligned_points = np.dot(transformation[:3, :3], source.points.T).T + transformation[:3, 3]

    print(f"Fine alignment complete. Final transformation:\n{transformation}")
    return PointCloud(aligned_points)

# The main function remains the same as in the previous artifact