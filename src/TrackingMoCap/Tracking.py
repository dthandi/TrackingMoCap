import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from scipy import stats as st
import time

# ---------------------------------------------------
# Label Stationary Markers
# ---------------------------------------------------



# 1) Find the speeds and distances of the all the markers between frames
#       a) OPTIONAL: If distance is less than a threshold, the marker is likely the same as the previous frame
#       b) If the speed is below a certain threshold, the marker is stationary
#       c) Temporarily label all markers after landing as non-stationary to prevent labelling the perched bird
# 3) Cluster the stationary markers
#       a) Check how long each cluster persists in the data and remove short duration clusters
# 5) Find the most common position within clusters
#       a) Only look at frames where the bird is flying
#       b) For each stationary cluster, find most common position (modal)
#       c) OPTIONAL: Label outliers too far from this position as 0
#       d) Find and label markers (all frames) close to the most common cluster position
# 6) OPTIONAL: Remove stationary markers outside an area of interest 
# ---------------------------------------------------

def label_stationary(marker_coordinates, frame_identifiers, threshold_speed=0.015, only_before_frame=230, area_of_interest=None):

    """
    Labels markers as stationary or not based on their displacement over a range of frames.

    Args:
    marker_table: DataFrame containing 'XYZ' coordinates and 'frame' numbers.
    threshold_proximity: Float, the threshold for determining if a marker is stationary.

    Returns:
    stationary_labels: Series indicating whether each marker is stationary (1) or not (0).
    """


    # # Initialize labels and set stationary markers
    stationary_labels = np.zeros(len(frame_identifiers), dtype=int)
    
    # Calculate the distances and speeds of all markers between frames 
    indices_, distances, speeds = calculate_movement_between_frames(frame_identifiers, 
                                                                    marker_coordinates, 
                                                                    consecutive_frames=False, 
                                                                    frame_gap=1, 
                                                                    only_before_frame=only_before_frame)
    start_indices, end_indices = indices_
    
    # ------------------------------------------------------------------------
    # --- Optional: Use Distance Between Frames to find stationary markers ---


    # If distance is less than 1mm, the marker is the same
    # is_same_marker = distances < 0.2

    # Match the close distance indices to the frame identifiers. Label markers below
    # the distance threshold as stationary markers.
    # same_marker_indices = np.unique(np.concatenate([start_indices[is_same_marker], end_indices[is_same_marker]]))
    # stationary_labels[same_marker_indices] = 1

    # ------------------------------------------------------------------------
    # --- Use Speed Between Frames to find stationary markers ----------------

    # Use a threshold speed to determine if a marker is stationary    

    is_stationary = abs(speeds) < threshold_speed

    # Match the speed indices to the frame identifiers. Label markers below 
    # the speed threshold as stationary markers
    stationary_indices = np.unique(np.concatenate([start_indices[is_stationary], end_indices[is_stationary]]))
    stationary_labels[stationary_indices] = 1


    if only_before_frame is not None:
        stationary_labels[frame_identifiers > only_before_frame] = 0

    # ------------------------------------------------------------------------
    # --- Cluster the stationary markers -------------------------------------
    
    # - first check there is more than one point to cluster
    # - increases the number of clusters until the Calinski-Harabasz score is maximized
    # - assigns the cluster labels to the stationary labels
    

    if np.sum(stationary_labels) > 1:
        stationary_labels = cluster_labels(marker_coordinates, stationary_labels)

    # ------------------------------------------------------------------------
    # --- Check how long each cluster persists in the data--------------------

    # - if a cluster lasts for less than 75% of the frames, label it as non stationary

    stationary_labels = remove_short_duration_clusters(frame_identifiers, marker_coordinates, stationary_labels, 0.75)

    # ------------------------------------------------------------------------
    # --- Find most common position within clusters --------------------------

    # - For each stationary cluster, find most common position
    # - Best to choose most common only when the bird is flying -- and not near the perch
    # - Label outliers too far from this position as 0


    label_mode = find_most_common_stationary_position(marker_coordinates, 
                                                     frame_identifiers, 
                                                     stationary_labels,
                                                     only_before_frame=only_before_frame)
    
    # ------------------------------------------------------------------------
    # --- Optional: Exclude stationary markers too far away ----------------------------

    stationary_labels = label_far_markers(marker_coordinates, 
                                          stationary_labels, 
                                          label_mode, 
                                          threshold=20)

    
    # ------------------------------------------------------------------------
    # --- Find markers close to most common cluster position -----------------

    stationary_labels = find_close_markers(marker_coordinates, 
                                           stationary_labels, 
                                           label_mode, 
                                           threshold=10)
        

    # ------------------------------------------------------------------------
    # --- Optional: Exclude stationary markers outside area of interest ----------------

    # if area_of_interest is not None:
    #     x_min, x_max, y_min, y_max, z_min, z_max = area_of_interest

    #     # Find markers outside the area of interest that are labelled as stationary
    #     # and set their labels to 0
    #     outside_area = np.where(
    #        (marker_coordinates[:, 0] < x_min) | (marker_coordinates[:, 1] > x_max) |
    #        (marker_coordinates[:, 1] < y_min) | (marker_coordinates[:, 1] > y_max) |
    #        (marker_coordinates[:, 2] < z_min) | (marker_coordinates[:, 2] > z_max)
    #         )[0]
    #     stationary_outside_indices = outside_area[stationary_labels[outside_area] != 0]
    #     stationary_labels[stationary_outside_indices] = 0

    return stationary_labels


# ------------------------------------------------------------------------
# --- Helper Functions ---------------------------------------------------
# ------------------------------------------------------------------------

def calculate_movement_between_frames(frame_ids, marker_coords, return_unit_vectors=False, consecutive_frames=True, frame_gap=None, only_before_frame=None):
    """
    Calculate the distances and speeds between frames based on marker coordinates.
    
    Args:
    frame_ids: A numpy array of frame identifiers.
    marker_coords: A numpy array of marker coordinates in a three-dimensional space.
    return_unit_vectors: A boolean flag to return unit vectors representing the direction and magnitude of displacement between markers.
    
    Returns:
    distances: A numpy array of Euclidean distances between markers.
    speeds: A numpy array of speeds between markers.
    unit_displacement_vectors: (Optional) A numpy array of unit vectors representing the direction and magnitude of displacement between markers.
    
    Optional:
    If return_unit_vectors is set to False, the function will return distances and speeds.
    If return_unit_vectors is set to True, the function will return distances, speeds, and unit_displacement_vectors.
    """
    
    
    

    # If consecutive_frames is set to True, calculate the speeds between consecutive frames
    if consecutive_frames:
        unique_frames, indices = np.unique(frame_ids, return_index=True)
        frame_durations = np.diff(unique_frames)
        start_indices = indices[:-1]
        end_indices = indices[1:]

    # If frame_gap is specified, calculate the speeds between frames with the specified gap
    else:
        start_indices, end_indices = find_edges(frame_ids, frame_gap)
        frame_durations = frame_ids[end_indices] - frame_ids[start_indices]

    # Ensure end_indices and start_indices are valid for accessing marker_coords
    if np.any(end_indices >= len(marker_coords)) or np.any(start_indices >= len(marker_coords)):
        raise ValueError("Index out of bounds. Check frame ID continuity and marker coordinates alignment.")



    # Calculate displacements and distances
    displacement_vectors = marker_coords[end_indices] - marker_coords[start_indices]
    distances = np.linalg.norm(displacement_vectors, axis=1)

    # Compute speeds
    speeds = distances / frame_durations

    # Compute unit vectors if specified
    if return_unit_vectors:
        unit_displacement_vectors = displacement_vectors / distances[:, np.newaxis]
        return (start_indices, end_indices), distances, speeds, unit_displacement_vectors

    return (start_indices, end_indices), distances, speeds

def find_edges(frame_identifiers, frame_gap):
    """
    Helper function to find index pairs with a specified gap.
    
    Args:
    frame_identifiers: A numpy array of frame identifiers.
    frame_gap: An integer representing the gap between frames.
    
    Returns:
    start_frame_indices: Indices of the starting frames.
    end_frame_indices: Indices of the ending frames.
    """
    # Create an array of all differences between frame identifiers
    frame_diff = np.abs(np.subtract.outer(frame_identifiers, frame_identifiers))

    # Find indices where the difference matches the frame gap
    start_frame_indices, end_frame_indices = np.where(frame_diff == frame_gap)

    # Ensure starting index is less than ending index
    valid_indices = start_frame_indices < end_frame_indices
    start_frame_indices = start_frame_indices[valid_indices]
    end_frame_indices = end_frame_indices[valid_indices]

    return start_frame_indices, end_frame_indices

def cluster_labels(marker_coordinates, stationary_labels):

    """
    Cluster stationary markers based on their coordinates.

    Args:
    marker_coordinates: A numpy array of marker coordinates in a three-dimensional space.
    stationary_labels: A numpy array of labels indicating whether each marker is stationary.

    Returns:
    stationary_labels: A numpy array of labels indicating the cluster each marker belongs to.

    """

    updated_labels = np.copy(stationary_labels)
    stationary_coords = marker_coordinates[updated_labels == 1]
    
    num_stationary = len(stationary_coords)
    k_range = range(2, min(5, num_stationary))

    if len(k_range) > 1:
            scores = [calinski_harabasz_score(stationary_coords, KMeans(n_clusters=k).fit_predict(stationary_coords)) for k in k_range]
            optimal_k = k_range[np.argmax(scores)]
            kmeans = KMeans(n_clusters=optimal_k)
            cluster_labels = kmeans.fit_predict(stationary_coords) + 1  # offset by 1 to avoid zero label
            updated_labels[updated_labels == 1] = cluster_labels

    return updated_labels

def remove_short_duration_clusters(frame_identifiers, marker_coordinates, stationary_labels, min_percent_frames):
    """
    Remove clusters that persist for less than a specified percentage of frames.
    
    Args:
    frame_identifiers: A numpy array of frame identifiers.
    marker_coordinates: A numpy array of marker coordinates in a three-dimensional space.
    stationary_labels: A numpy array of labels indicating whether each marker is stationary.
    min_percent_frames: A float representing the minimum percentage of frames a cluster must persist to remain labelled.
    
    Returns:
    stationary_labels: A numpy array of labels indicating the cluster each marker belongs to.
    """

    updated_labels = np.copy(stationary_labels)

    unique_frames = np.unique(frame_identifiers)
    for label in np.unique(updated_labels):
        if label == 0:
            continue
        label_indices = np.where(updated_labels == label)[0]
        label_coords = marker_coordinates[label_indices]

        # We only want to calculate the length without nans
        num_frames = len(np.unique(label_coords[~np.isnan(label_coords)]))


        if num_frames < min_percent_frames * len(unique_frames):
            updated_labels[label_indices] = 0
    
    return updated_labels

def find_most_common_stationary_position(marker_coordinates, frame_identifiers, stationary_labels, is_flying_frames = None, threshold=10, only_before_frame=None):

    """
    Find the most common position of each cluster when the bird is flying and label outliers as 0.

    Args:
    marker_coordinates: A numpy array of marker coordinates in a three-dimensional space.
    frame_identifiers: A numpy array of frame identifiers.
    stationary_labels: A numpy array of labels indicating whether each marker is stationary and the cluster. 
    only_before_frame: If known, representing the last frame where the bird is flying.

    Returns:
    label_mode: A numpy array of the mode position of each cluster when the bird is flying.
    stationary_labels: A numpy array of labels indicating the cluster each marker belongs to.
    """

    # Initialize an array to store the mode position of each cluster
    label_mode = {}

    # For each cluster, find the mode position when the bird is flying
    for label in np.unique(stationary_labels):
        if label == 0:
            continue
        label_coords = marker_coordinates[stationary_labels == label]
        label_frames = frame_identifiers[stationary_labels == label]

        # If the flying frames are known, set the valid indices as the flying frames
        if is_flying_frames is not None:
            valid_indices = np.isin(label_frames, is_flying_frames)

        # If the last frame is known, set the last frame as the maximum frame
        elif only_before_frame is not None:
            valid_indices = label_frames < only_before_frame
        
        # Otherwise, set all indices as valid
        else:
            valid_indices = np.ones(len(label_frames), dtype=bool)

        # label_coords = label_coords[valid_indices]
        # label_frames = label_frames[valid_indices]

        # Find the mode position of each cluster when the bird is flying
        # Scale and round the coordinates for 1cm precision
        rounded_coords = np.round(label_coords[valid_indices]).astype(int)
    
        # Find the mode position of each cluster
        mode_coords = st.mode(rounded_coords, axis=0, keepdims=True, nan_policy='omit').mode[0]

        # Store the mode position of each cluster in original scale
        label_mode[label] = mode_coords

    return label_mode


def label_far_markers(marker_coordinates, stationary_labels, label_mode, threshold=10):

    """
    Find markers too far to the mode position of each cluster and remove their label.

    Args:
    marker_coordinates: A numpy array of marker coordinates in a three-dimensional space.
    stationary_labels: A numpy array of labels indicating the cluster each marker belongs to.
    label_mode: A dictionary of the mode position of each cluster when the bird is flying.
    threshold: A float representing the threshold distance for determining if a marker is close to the mode position.

    Returns:
    stationary_labels: A numpy array of labels indicating the cluster each marker belongs to.
    """

    # Copy labels to avoid changing the original array during iteration
    updated_labels = np.copy(stationary_labels)
    

    for label, position in label_mode.items():
        if label == 0:
            continue
        
        # Calculate the distance in xy between the marker coordinates and the mode position (ignore z)
        distances = np.linalg.norm(marker_coordinates - position, axis=1)
        
        nearby_points_indices = np.where(distances > threshold)[0]
        updated_labels[nearby_points_indices] = 0
        
    return updated_labels





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
    
    displacement_vectors = marker_coordinates[end_frame_indices, :] - marker_coordinates[start_frame_indices, :]
    
    marker_distances = np.linalg.norm(displacement_vectors, axis=1)

    if return_unit_vectors:
        unit_displacement_vectors = displacement_vectors / marker_distances[:, np.newaxis]
        return marker_distances, start_frame_indices, end_frame_indices, unit_displacement_vectors
    return marker_distances, start_frame_indices, end_frame_indices

def find_close_markers(marker_coordinates, stationary_labels, label_mode, threshold=10):

    """
    Find markers close to the mode position of each cluster and assign the same label.

    Args:
    marker_coordinates: A numpy array of marker coordinates in a three-dimensional space.
    stationary_labels: A numpy array of labels indicating the cluster each marker belongs to.
    label_mode: A numpy array of the mode position of each cluster when the bird is flying.
    threshold: A float representing the threshold distance for determining if a marker is close to the mode position.

    Returns:
    stationary_labels: A numpy array of labels indicating the cluster each marker belongs to.
    """

    # Copy labels to avoid changing the original array during iteration
    updated_labels = np.copy(stationary_labels)
    

    for label, position in label_mode.items():
        if label == 0:
            continue
        
        # Calculate the distance in xy between the marker coordinates and the mode position (ignore z)
        distances = np.linalg.norm(marker_coordinates - position, axis=1)
        
        nearby_points_indices = np.where(distances < threshold)[0]
        updated_labels[nearby_points_indices] = label
        
    return updated_labels

def estimate_moving_frames(marker_coordinates, frame_identifiers, only_before_frame = None, min_speed=2, max_speed=100):

    """
    Estimate the frames where the bird is moving based on the speed of the markers.
        - Calculate the speed of the markers by finding the centroid for each frame
        - Calculate the distance between the centroids across frames
        - Find the frames where the bird is probably moving based on the speed of the markers

    Args:
    marker_coordinates: A numpy array of marker coordinates in a three-dimensional space.
    frame_identifiers: A list or numpy array of frame identifiers corresponding to the marker coordinates.
    last_frame: If known, representing the last frame where the bird is flying.
    min_speed: A float representing the minimum speed threshold for determining if the bird is moving.
    max_speed: A float representing the maximum speed threshold for determining if the bird is moving.

    Returns:
    flying_frames: A numpy array of frame identifiers where the bird is moving.

    """


    # Calculate the speed of the markers by finding the centroid for each frame
    # and calculating the distance between the centroids across frames
    unique_frames, inverse_indices = np.unique(frame_identifiers, return_inverse=True)

    # Initialise an array to hold centroids
    centroids = np.zeros((len(unique_frames), marker_coordinates.shape[1]))
    
    # Calculate centroids for each unique frame
    for ii in range(len(unique_frames)):
        centroids[ii] = np.mean(marker_coordinates[inverse_indices == ii], axis=0)

    # Create a moving average of the centroids using 10 frames
    #    - centroids has shape (n, 3) so need to apply convolution to each dimension
    centroids = np.vstack([np.convolve(centroids[:, ii], np.ones(10) / 10, mode='same') for ii in range(marker_coordinates.shape[1])]).T

    # Calculate distances between frames of the centroids
    # frame_durations = np.diff(unique_frames) # In case there are missing frames
    # distances = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    # speeds = distances / frame_durations

    indices, distances, speeds = calculate_movement_between_frames(unique_frames, centroids)

    # Find the frames where the bird is probably moving
    is_flying = np.where((speeds > min_speed) & (speeds < max_speed))[0]

    # Match the speed indices to the frame identifiers
    start_indices, end_indices = indices
    is_flying_frames = np.unique(np.concatenate([unique_frames[start_indices[is_flying]], unique_frames[end_indices[is_flying]]]))
    
    # If the last frame is known, set the last frame as the maximum frame
    if only_before_frame is not None:
        is_flying_frames = is_flying_frames[is_flying_frames <= only_before_frame]

    return is_flying_frames
