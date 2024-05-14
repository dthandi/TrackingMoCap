import numpy as np

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



