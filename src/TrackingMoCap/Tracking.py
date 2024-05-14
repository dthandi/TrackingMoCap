import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial.distance import cdist
from scipy.stats import norm


def label_stationary(marker_coordinates, frame_identifiers, threshold_proximity=0.001):

    """
    Labels markers as stationary or not based on their displacement over a range of frames.

    Args:
    marker_table: DataFrame containing 'XYZ' coordinates and 'frame' numbers.
    threshold_proximity: Float, the threshold for determining if a marker is stationary.

    Returns:
    stationary_labels: Series indicating whether each marker is stationary (1) or not (0).
    """
        
    # Calculate distances and indices over frame gaps from 1 to 10
    distances, start_indices, end_indices = compute_marker_distances(marker_coordinates,
                                                                    frame_identifiers,
                                                                    frame_gap=list(range(1, 11)))

    # Normalize distances by frame durations and check against the threshold
    frame_durations = frame_identifiers[end_indices] - frame_identifiers[start_indices]
    normalized_distances = distances / frame_durations
    is_stationary = normalized_distances < threshold_proximity

    # Initialize labels and set stationary markers
    stationary_labels = np.zeros(len(frame_identifiers), dtype=int)
    stationary_indices = np.unique(np.concatenate((start_indices[is_stationary], end_indices[is_stationary])))
    stationary_labels[stationary_indices] = 1

    if np.sum(stationary_labels) > 1:  # Ensure there are at least two points to cluster
        stationary_coords = marker_coordinates[stationary_labels == 1]
        num_stationary = len(stationary_coords)
        k_range = range(2, min(10, num_stationary))  # Ensure valid range for k (at least 2, at most num_stationary - 1)

        if len(k_range) > 1:
            scores = [calinski_harabasz_score(stationary_coords, KMeans(n_clusters=k).fit_predict(stationary_coords)) for k in k_range]
            optimal_k = k_range[np.argmax(scores)]
            kmeans = KMeans(n_clusters=optimal_k)
            cluster_labels = kmeans.fit_predict(stationary_coords) + 1  # offset by 1 to avoid zero label
            stationary_labels[stationary_labels == 1] = cluster_labels

    return stationary_labels




def compute_marker_distances(marker_coordinates, 
                             frame_identifiers, 
                             frame_gap=0, 
                             return_unit_vectors=False):
    """
    Compute distances between based on their coordinates, 
    frame identifiers, and specified frame gap.

    Args:
    marker_coordinates: A numpy array of marker coordinates in a three-dimensional space.
    frame_identifiers: A list or numpy array of frame identifiers corresponding to the marker coordinates.
    frame_gap: An integer representing the desired gap between frames for calculating distances.
    return_unit_vectors: A boolean flag to return unit vectors representing the direction and magnitude of displacement between markers.
    
    Returns:
    marker_distances: A numpy array of Euclidean distances between markers.
    start_frame_indices: Indices of the starting frames for the distances computed.
    end_frame_indices: Indices of the ending frames for the distances computed.
    unit_displacement_vectors: Unit vectors representing the direction and magnitude of displacement between markers.
    Optional:
    If return_unit_vectors is set to False, the function will return marker_distances, start_frame_indices, and end_frame_indices.
    If return_unit_vectors is set to True, the function will return marker_distances, start_frame_indices, end_frame_indices, and unit_displacement_vectors.
    """
    
    start_frame_indices, end_frame_indices = find_edges(frame_identifiers, frame_gap)
    
    displacement_vectors = marker_coordinates[end_frame_indices, :3] - marker_coordinates[start_frame_indices, :3]
    
    marker_distances = np.linalg.norm(displacement_vectors, axis=1)

    if return_unit_vectors:
        unit_displacement_vectors = displacement_vectors / marker_distances[:, np.newaxis]
        return marker_distances, start_frame_indices, end_frame_indices, unit_displacement_vectors
    return marker_distances, start_frame_indices, end_frame_indices


def find_edges(frame_identifiers, frame_gap):

    """
    Helper function to find the index pairs with a specified gap. 
    
    Args:
    frame_identifiers: A list or numpy array of frame identifiers.
    frame_gap: An integer representing the gap between frames.
    
    Returns:
    starting_frame_indices: Indices of the starting frames.
    ending_frame_indices: Indices of the ending frames.

    """

    num_frames = len(frame_identifiers)
    frame_diff = np.array(frame_identifiers).reshape(-1, 1) - np.array(frame_identifiers)

    # Check if frame_gap is a list or array, and handle accordingly
    frame_match_mask = np.isin(frame_diff, frame_gap)

    start_frame_indices, end_frame_indices = np.where(frame_match_mask)
    valid_indices = start_frame_indices < end_frame_indices
    start_frame_indices = start_frame_indices[valid_indices]
    ending_frame_indices = end_frame_indices[valid_indices]

    return start_frame_indices, ending_frame_indices



