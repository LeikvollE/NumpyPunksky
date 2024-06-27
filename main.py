import numpy as np
import laspy
import time
from scipy.spatial import cKDTree
import multiprocessing as mp
from functools import partial
import pyvista as pv

from PointCloud import PointCloud
from alignment import fine_align_point_clouds


def load_las_as_point_cloud(filename: str) -> PointCloud:
    try:
        las = laspy.read(filename)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        return PointCloud(points)
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        raise


def process_chunk(chunk, nb_neighbors, std_ratio):
    tree = cKDTree(chunk)
    distances, _ = tree.query(chunk, k=nb_neighbors)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    std_dev = np.std(mean_distances)
    threshold = mean_distances.mean() + std_ratio * std_dev
    mask = mean_distances < threshold
    return chunk[mask]


def parallel_statistical_outlier_removal(pcd: PointCloud, nb_neighbors: int = 20, std_ratio: float = 2.0,
                                         n_jobs: int = -1) -> PointCloud:
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    chunk_size = len(pcd.points) // n_jobs
    chunks = [pcd.points[i:i + chunk_size] for i in range(0, len(pcd.points), chunk_size)]

    with mp.Pool(n_jobs) as pool:
        processed_chunks = pool.map(partial(process_chunk, nb_neighbors=nb_neighbors, std_ratio=std_ratio), chunks)

    return PointCloud(np.vstack(processed_chunks))


def voxel_downsample(pcd: PointCloud, voxel_size: float = 0.05) -> PointCloud:
    voxel_indices = np.floor(pcd.points / voxel_size).astype(int)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    return PointCloud(pcd.points[unique_indices])


def preprocess_point_cloud(pcd: PointCloud, voxel_size: float = 0.05) -> PointCloud:
    pcd_filtered = parallel_statistical_outlier_removal(pcd)
    print(f"After SOR filtering: {len(pcd_filtered.points)} points")
    pcd_down = voxel_downsample(pcd_filtered, voxel_size)
    return pcd_down


def parallel_detect_changes(source_pcd: PointCloud, target_pcd: PointCloud, threshold: float = 0.05,
                            n_jobs: int = -1) -> np.ndarray:
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    tree = cKDTree(target_pcd.points)

    chunk_size = len(source_pcd.points) // n_jobs
    chunks = [source_pcd.points[i:i + chunk_size] for i in range(0, len(source_pcd.points), chunk_size)]

    with mp.Pool(n_jobs) as pool:
        distances_chunks = pool.map(partial(tree.query, k=1), chunks)

    distances = np.concatenate([chunk[0] for chunk in distances_chunks])
    changed_indices = distances > threshold
    changed_points = source_pcd.points[changed_indices]

    print(f"Distance statistics:")
    print(f"  Min distance: {np.min(distances)}")
    print(f"  Max distance: {np.max(distances)}")
    print(f"  Mean distance: {np.mean(distances)}")
    print(f"  Median distance: {np.median(distances)}")
    print(f"  Standard deviation: {np.std(distances)}")
    print(f"Threshold: {threshold}")
    print(f"Number of points above threshold: {np.sum(changed_indices)}")

    return changed_points


def visualize_changes(source_pcd: PointCloud, target_pcd: PointCloud, changed_points: np.ndarray,
                      sample_size: int = 20000000) -> None:
    plotter = pv.Plotter()

    source_sample = source_pcd.points[
        np.random.choice(len(source_pcd.points), min(sample_size, len(source_pcd.points)), replace=False)]
    target_sample = target_pcd.points[
        np.random.choice(len(target_pcd.points), min(sample_size, len(target_pcd.points)), replace=False)]
    changed_sample = changed_points[
        np.random.choice(len(changed_points), min(sample_size, len(changed_points)), replace=False)]

    source_cloud = pv.PolyData(source_sample)
    target_cloud = pv.PolyData(target_sample)
    changed_cloud = pv.PolyData(changed_sample)

    plotter.add_mesh(source_cloud, color='red', point_size=2, render_points_as_spheres=True, label='Source')
    plotter.add_mesh(target_cloud, color='green', point_size=2, render_points_as_spheres=True, label='Target')
    #plotter.add_mesh(changed_cloud, color='blue', point_size=3, render_points_as_spheres=True, label='Changes')

    plotter.add_legend()
    plotter.show()


def main() -> None:
    try:
        start_time = time.time()

        source_pcd = load_las_as_point_cloud("clouds/ST5300_10_laz1_4.laz")
        target_pcd = load_las_as_point_cloud("clouds/ST5300_12_laz1_4.laz")

        print(f"Loaded source point cloud with {len(source_pcd.points)} points")
        print(f"Loaded target point cloud with {len(target_pcd.points)} points")

        source_pcd_processed = preprocess_point_cloud(source_pcd, voxel_size=0.1)
        target_pcd_processed = preprocess_point_cloud(target_pcd, voxel_size=0.1)

        print(f"Preprocessed source point cloud has {len(source_pcd_processed.points)} points")
        print(f"Preprocessed target point cloud has {len(target_pcd_processed.points)} points")

        print("Performing fine alignment of point clouds...")
        aligned_source_pcd = source_pcd_processed#fine_align_point_clouds(source_pcd_processed, target_pcd_processed)
        print("Fine alignment complete")

        threshold = 0.3  # Adjust this value based on the scale of your point clouds
        changed_points = parallel_detect_changes(aligned_source_pcd, target_pcd_processed, threshold=threshold)

        print(f"Detected {len(changed_points)} changed points")

        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")

        visualize_changes(aligned_source_pcd, target_pcd_processed, changed_points)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()