# -*- coding: utf-8 -*-
"""
md_analysis_tools.py

A collection of Python functions for analyzing molecular dynamics (MD)
trajectories using MDAnalysis and other common scientific Python libraries.

This module provides tools for calculating:
- Backbone dihedral angles (Phi, Psi)
- Cartesian Principal Components (PCA)
- KMeans clustering results
- Representative frame indices from clusters or KDE minima
- 2D Kernel Density Estimates (KDE) and landscape minima
- Pairwise distances between atom groups
- Root Mean Square Fluctuations (RMSF)
- Radius of Gyration (Rg)
- Root Mean Square Deviation (RMSD) relative to a reference structure

Notes:
- Assumes standard PDB/trajectory formats readable by MDAnalysis.
- Some functions rely on optional dependencies (scikit-learn, scipy, scikit-image).
- Error handling is basic; robust production use might require more checks.
- Functions primarily return pandas DataFrames or NumPy arrays for easy downstream use.
"""

# --- Standard Library Imports ---
import os
import sys
import warnings

# --- Third-party Imports ---
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals, pca, align, rms
from MDAnalysis.exceptions import NoDataError

# Optional imports - checked before use
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not found. Clustering functionality will be unavailable.", ImportWarning)

try:
    from scipy import stats
    from scipy.spatial.distance import cdist
    from scipy.ndimage import gaussian_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    warnings.warn("SciPy not found. KDE and distance calculation functionality will be unavailable.", ImportWarning)

try:
    from skimage.feature import peak_local_max
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False
    # No warning here, as find_kde_minima_2d checks specifically

# --- Helper Function ---

def _check_file_exists(filepath, func_name):
    """Internal helper to check if a file exists and print error if not."""
    if not os.path.exists(filepath):
        print(f"Error in {func_name}: Input file not found: {filepath}", file=sys.stderr)
        return False
    return True

# --- Core Analysis Functions ---

def calculate_phi_psi_angles(universe, selection="protein", start_res=1, end_res=-1, start_frame=0, stop_frame=-1, step=1):
    """
    Calculates Phi and Psi dihedral angles for a specified residue range over a trajectory.

    Uses MDAnalysis.analysis.dihedrals.Ramachandran.

    Args:
        universe (MDAnalysis.Universe): An MDAnalysis Universe object containing topology and trajectory.
        selection (str, optional): Atom selection string for the protein/polymer containing the residues.
                                   Defaults to "protein".
        start_res (int, optional): The starting residue index (0-based) within the selection for dihedral calculation.
                                   Defaults to 1 (skipping the first residue which lacks phi).
        end_res (int, optional): The ending residue index (0-based, exclusive) within the selection.
                                 Defaults to -1 (skipping the last residue which lacks psi).
        start_frame (int, optional): First frame index to analyze. Defaults to 0.
        stop_frame (int, optional): Last frame index to analyze (exclusive).
                                    Defaults to -1 (until the end).
        step (int, optional): Step size for trajectory frames. Defaults to 1.

    Returns:
        pandas.DataFrame or None: A DataFrame containing Phi and Psi angles (in degrees)
                                  for each selected residue over the trajectory frames.
                                  Returns None if the analysis fails (e.g., no residues selected).
                                  Columns are named 'Phi_ResID', 'Psi_ResID' using actual residue IDs.
                                  An additional 'Frame' column indicates the trajectory frame index.
    """
    func_name = "calculate_phi_psi_angles"
    try:
        # Select the broader group first (e.g., protein)
        atom_group = universe.select_atoms(selection)
        if not atom_group:
            print(f"Warning in {func_name}: Selection '{selection}' resulted in 0 atoms.", file=sys.stderr)
            return None

        # Select residues *within* that group based on indices
        n_residues_in_group = len(atom_group.residues)
        actual_end_res = end_res if end_res != -1 else n_residues_in_group - 1 # Adjust default end

        # Check residue range validity relative to the selected group
        if start_res >= actual_end_res or start_res < 0 or actual_end_res > n_residues_in_group:
             print(f"Warning in {func_name}: Invalid residue index range [{start_res}:{actual_end_res}] for {n_residues_in_group} residues in selection '{selection}'.", file=sys.stderr)
             return None

        residues_to_analyze = atom_group.residues[start_res:actual_end_res]
        if not residues_to_analyze: # Check if the slice resulted in any residues
            print(f"Warning in {func_name}: No residues selected in the index range [{start_res}:{actual_end_res}].", file=sys.stderr)
            return None

        print(f"Info ({func_name}): Analyzing {len(residues_to_analyze)} residues (indices {start_res} to {actual_end_res-1} of selection '{selection}')")

        # Run Ramachandran analysis
        # Use a context manager for warnings if specific ones need suppression
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", category=Warning) # e.g., warnings.warn("Adding 180 degrees to phi")
        rama_analysis = dihedrals.Ramachandran(residues_to_analyze).run(
            start=start_frame, stop=stop_frame, step=step
        )

        angles_deg = rama_analysis.results.angles # Shape: (n_frames, n_residues, 2) [phi, psi]

        # Reshape and create DataFrame
        n_frames_analyzed = angles_deg.shape[0]
        n_residues_analyzed = angles_deg.shape[1]

        # Create column names like Phi_ResID, Psi_ResID using actual residue IDs
        columns = []
        analyzed_resid_ids = [res.resid for res in residues_to_analyze]
        for resid in analyzed_resid_ids:
            columns.append(f'Phi_{resid}')
            columns.append(f'Psi_{resid}')

        # Reshape the array: frames become rows, (phi/psi * residues) become columns
        reshaped_angles = angles_deg.reshape(n_frames_analyzed, n_residues_analyzed * 2)

        df_angles = pd.DataFrame(reshaped_angles, columns=columns)

        # Add Frame index column based on analysis range
        # Use the actual indices from the trajectory slice for accuracy
        analyzed_frame_indices = universe.trajectory.slice[start_frame:stop_frame:step].indices

        # Double check length consistency (should match n_frames_analyzed)
        if len(analyzed_frame_indices) != n_frames_analyzed:
            warnings.warn(f"Frame index length mismatch in {func_name}. Using simple arange.", RuntimeWarning)
            analyzed_frame_indices = np.arange(start_frame if start_frame is not None else 0,
                                                start_frame if start_frame is not None else 0 + n_frames_analyzed * step,
                                                step)[:n_frames_analyzed]

        df_angles.insert(0, 'Frame', analyzed_frame_indices)

        print(f"Success ({func_name}): Calculated Phi/Psi angles for {n_frames_analyzed} frames.")
        return df_angles

    except NoDataError as e:
         print(f"Error in {func_name}: MDAnalysis NoDataError - often due to issues with atom selections for dihedrals (e.g., missing atoms in a residue). {e}", file=sys.stderr)
         return None
    except AttributeError as e:
         print(f"Error in {func_name}: AttributeError - Possibly issue with residue selection or accessing properties. {e}", file=sys.stderr)
         return None
    except Exception as e:
         print(f"An unexpected error occurred in {func_name}: {e}", file=sys.stderr)
         return None


def perform_cartesian_pca(universe, select="name CA", align=True, align_selection=None,
                           n_components=None, mean_structure=None,
                           start_frame=0, stop_frame=-1, step=1,
                           in_memory_align=False):
    """
    Performs Cartesian Principal Component Analysis (PCA) on MD trajectory data.

    Uses MDAnalysis.analysis.pca.PCA and optionally MDAnalysis.analysis.align.AlignTraj.

    Args:
        universe (MDAnalysis.Universe): An MDAnalysis Universe object with topology and trajectory.
        select (str, optional): Atom selection string for the atoms to include in PCA.
                                Defaults to "name CA".
        align (bool, optional): Whether to perform RMSD alignment before PCA. Defaults to True.
        align_selection (str, optional): Atom selection string for the atoms to use for RMSD alignment.
                                      If None and align is True, defaults to the same as `select`.
                                      Defaults to None.
        n_components (int, optional): Number of principal components to calculate.
                                      If None, all components are calculated. Defaults to None.
        mean_structure (MDAnalysis.Universe or AtomGroup, optional): A structure to use as the mean
                                     for PCA instead of calculating the mean from the trajectory.
                                     Defaults to None (mean calculated from trajectory).
        start_frame (int, optional): First frame index to analyze. Defaults to 0.
        stop_frame (int, optional): Last frame index to analyze (exclusive).
                                    Defaults to -1 (until the end).
        step (int, optional): Step size for trajectory frames. Defaults to 1.
        in_memory_align (bool, optional): Load trajectory into memory for alignment. Can be faster
                                          for smaller trajectories but uses more RAM. Defaults to False.

    Returns:
        MDAnalysis.analysis.pca.PCA or None: The fitted PCA object containing results
                                             (components, variance, mean, transform method),
                                             or None if an error occurs.
    """
    func_name = "perform_cartesian_pca"
    aligned_universe = universe # Start with the original universe

    try:
        # --- Alignment Step (Optional) ---
        if align:
            if align_selection is None:
                align_select = select # Default to using the same selection for alignment and PCA
            else:
                align_select = align_selection

            # Check alignment selection validity
            if not universe.select_atoms(align_select):
                 raise ValueError(f"Alignment selection '{align_select}' resulted in 0 atoms.")

            print(f"Info ({func_name}): Performing alignment using selection: '{align_select}'")
            try:
                # Align to the first frame of the specified range if no mean_structure provided
                ref_frame_for_align = mean_structure if mean_structure is not None else universe
                if mean_structure is None:
                     # Ensure reference frame is within the analyzed slice if aligning to trajectory
                     ref_actual_frame_idx = start_frame if start_frame is not None else 0
                     ref_frame_for_align.trajectory[ref_actual_frame_idx] # Go to reference frame

                aligner = align.AlignTraj(universe, ref_frame_for_align,
                                          select=align_select,
                                          in_memory=in_memory_align,
                                          filename=None) # Align in memory
                aligner.run(start=start_frame, stop=stop_frame, step=step)
                # Universe is modified in place by AlignTraj when filename=None
                print(f"Info ({func_name}): Alignment completed.")
            except Exception as e:
                 print(f"Error in {func_name}: Alignment failed. {e}", file=sys.stderr)
                 return None
        else:
             print(f"Info ({func_name}): Skipping alignment.")

        # --- PCA Step ---
        # Check PCA selection validity on the (potentially aligned) universe
        if not aligned_universe.select_atoms(select):
             raise ValueError(f"PCA selection '{select}' resulted in 0 atoms.")

        print(f"Info ({func_name}): Performing PCA using selection: '{select}'")

        pca_analysis = pca.PCA(aligned_universe,
                               select=select,
                               align=False, # IMPORTANT: Alignment was done separately above
                               mean=mean_structure,
                               n_components=n_components)
        pca_analysis.run(start=start_frame, stop=stop_frame, step=step)

        print(f"Success ({func_name}): PCA completed.")
        # Provide summary of results
        actual_n_components = pca_analysis.n_components
        print(f"  Calculated {actual_n_components} principal components.")
        print(f"  Cumulative variance covered: {pca_analysis.cumulated_variance[actual_n_components-1]:.3f}")
        if actual_n_components >= 3:
             print(f"  Variance covered by first 3 PCs: {pca_analysis.cumulated_variance[2]:.3f}")

        return pca_analysis

    except ValueError as e: # Catch selection errors specifically
         print(f"Error in {func_name}: {e}", file=sys.stderr)
         return None
    except Exception as e:
         print(f"An unexpected error occurred in {func_name}: {e}", file=sys.stderr)
         return None


def perform_kmeans(data, n_clusters, scale_data=False, random_state=None, **kmeans_kwargs):
    """
    Performs KMeans clustering on the input data.

    Uses scikit-learn's KMeans implementation. Requires scikit-learn to be installed.

    Args:
        data (numpy.ndarray or pandas.DataFrame): The input data for clustering.
                                                 Assumes rows are samples and columns are features.
        n_clusters (int): The number of clusters (k) to form.
        scale_data (bool, optional): Whether to standardize the data (mean 0, variance 1)
                                     before clustering. Recommended if features have
                                     different scales. Defaults to False.
        random_state (int, optional): Determines random number generation for centroid
                                      initialization. Use an int for reproducible results.
                                      Defaults to None.
        **kmeans_kwargs: Additional keyword arguments to pass directly to
                         sklearn.cluster.KMeans (e.g., init='k-means++', n_init=10).

    Returns:
        tuple or None: A tuple containing:
                       - kmeans_model (sklearn.cluster.KMeans): The fitted KMeans object.
                       - labels (numpy.ndarray): Array of cluster labels for each data point.
                       - centroids (numpy.ndarray): Coordinates of the cluster centers.
                       - data_used (numpy.ndarray): The actual data used for clustering
                         (scaled if scale_data=True).
                       Returns None if scikit-learn is unavailable or an error occurs.
    """
    func_name = "perform_kmeans"
    if not _SKLEARN_AVAILABLE:
        print(f"Error in {func_name}: scikit-learn is required for KMeans clustering.", file=sys.stderr)
        return None

    try:
        # Ensure data is a NumPy array
        if isinstance(data, pd.DataFrame):
            data_np = data.values
        elif isinstance(data, np.ndarray):
            data_np = data
        else:
            raise TypeError("Input 'data' must be a NumPy array or pandas DataFrame.")

        if data_np.ndim != 2:
             raise ValueError(f"Input 'data' must be 2-dimensional (samples, features), got shape {data_np.shape}")
        if data_np.shape[0] < n_clusters:
             raise ValueError(f"Number of samples ({data_np.shape[0]}) must be >= number of clusters ({n_clusters}).")

        data_to_cluster = data_np
        scaler = None # To potentially inverse transform centroids later if needed

        # Optional Scaling
        if scale_data:
            print(f"Info ({func_name}): Scaling data before clustering.")
            scaler = StandardScaler()
            data_to_cluster = scaler.fit_transform(data_to_cluster)
            # Note: Centroids returned will be in the scaled space.

        # --- KMeans Clustering ---
        print(f"Info ({func_name}): Performing KMeans clustering with k={n_clusters}.")

        # Sensible defaults for KMeans if not overridden
        kmeans_params = {"init": "k-means++", "n_init": 10}
        kmeans_params.update(kmeans_kwargs) # Overwrite defaults with user kwargs

        kmeans_model = KMeans(n_clusters=n_clusters,
                              random_state=random_state,
                              **kmeans_params)
        # Fitting calculates labels and centroids
        kmeans_model.fit(data_to_cluster)

        labels = kmeans_model.labels_
        centroids = kmeans_model.cluster_centers_ # These are in scaled space if scale_data=True

        print(f"Success ({func_name}): KMeans clustering completed.")
        print(f"  Inertia (SSE): {kmeans_model.inertia_:.3f}")

        # Note: We return the 'data_used' (potentially scaled) as it corresponds directly
        # to the 'labels' and 'centroids' returned.
        return kmeans_model, labels, centroids, data_to_cluster

    except Exception as e:
         print(f"An unexpected error occurred in {func_name}: {e}", file=sys.stderr)
         return None


def find_closest_frames_to_centroids(data, labels, centroids):
    """
    Finds the index of the data point closest to each cluster centroid.

    Useful for extracting representative structures/frames from clusters based on
    the data space used for clustering (e.g., PCA projections, distances).

    Args:
        data (numpy.ndarray): The original (or scaled) data that was clustered,
                              shape (n_samples, n_features).
        labels (numpy.ndarray): Cluster labels for each data point, shape (n_samples,).
        centroids (numpy.ndarray): Cluster centroids, shape (n_clusters, n_features).
                                  Should be in the same space as `data`.

    Returns:
        list or None: A list where each element is the index (relative to the input `data` array)
                      of the data point closest to the corresponding centroid. Index corresponds
                      to the cluster ID (0, 1, 2...). Contains None for empty clusters.
                      Returns None if SciPy is unavailable or an error occurs.
    """
    func_name = "find_closest_frames_to_centroids"
    if not _SCIPY_AVAILABLE:
        print(f"Error in {func_name}: SciPy is required for distance calculations (cdist).", file=sys.stderr)
        return None

    try:
        n_clusters = centroids.shape[0]
        closest_indices = [None] * n_clusters # Initialize with None

        # Pre-calculate mapping from label to indices for efficiency
        label_indices_map = {label: np.where(labels == label)[0] for label in range(n_clusters)}

        for i in range(n_clusters):
            cluster_point_indices = label_indices_map.get(i) # Get original indices for cluster i

            # Handle potentially empty clusters
            if cluster_point_indices is None or len(cluster_point_indices) == 0:
                 print(f"Warning in {func_name}: Cluster {i} has no points assigned.", file=sys.stderr)
                 continue # Keep closest_indices[i] as None

            # Select the data points belonging to this cluster
            cluster_points = data[cluster_point_indices]

            # Calculate distances from points in this cluster to the cluster's centroid
            # cdist expects arrays of points: (n_points, n_dims) and (1, n_dims) for single centroid
            distances = cdist(cluster_points, centroids[i].reshape(1, -1), 'euclidean')

            # Find the index *within the subset cluster_points array* of the minimum distance
            min_dist_index_in_subset = np.argmin(distances)

            # Map this subset index back to the index in the *original data array*
            original_data_index = cluster_point_indices[min_dist_index_in_subset]
            closest_indices[i] = original_data_index

            # Optional: Print info
            # print(f"Info ({func_name}): Cluster {i}: Closest Original Index={original_data_index}")

        print(f"Success ({func_name}): Found closest frame indices for {n_clusters} centroids.")
        return closest_indices

    except Exception as e:
         print(f"An unexpected error occurred in {func_name}: {e}", file=sys.stderr)
         return None


def calculate_kde_2d(x_data, y_data, grid_size=100j, bandwidth=None):
    """
    Calculates a 2D Kernel Density Estimate (KDE) on a grid using scipy.stats.

    Requires SciPy to be installed.

    Args:
        x_data (numpy.ndarray): 1D array of x-coordinates (e.g., PC1, distance 1).
        y_data (numpy.ndarray): 1D array of y-coordinates (e.g., PC2, distance 2).
        grid_size (complex): Number of grid points in each dimension (e.g., 100j for 100 points).
        bandwidth (float or str, optional): Bandwidth method for KDE ('scott', 'silverman',
                                           or a scalar value). Defaults to None (scipy default).

    Returns:
        tuple or None: A tuple containing:
                       - Z (numpy.ndarray): The KDE values on the grid.
                       - X (numpy.ndarray): The X coordinates of the grid mesh.
                       - Y (numpy.ndarray): The Y coordinates of the grid mesh.
                       Returns None if SciPy is unavailable or an error occurs.
    """
    func_name = "calculate_kde_2d"
    if not _SCIPY_AVAILABLE:
        print(f"Error in {func_name}: SciPy is required for KDE calculation.", file=sys.stderr)
        return None
    try:
        if len(x_data) != len(y_data):
            raise ValueError("Input x_data and y_data must have the same length.")
        if len(x_data) == 0:
             raise ValueError("Input data arrays are empty.")

        xmin, xmax = x_data.min(), x_data.max()
        ymin, ymax = y_data.min(), y_data.max()

        # Create grid points using np.mgrid
        X, Y = np.mgrid[xmin:xmax:grid_size, ymin:ymax:grid_size]
        positions = np.vstack([X.ravel(), Y.ravel()])

        # Stack input data for gaussian_kde
        values = np.vstack([x_data, y_data])

        # Compute KDE
        print(f"Info ({func_name}): Calculating KDE with bandwidth='{bandwidth if bandwidth else 'scipy_default'}'...")
        kernel = stats.gaussian_kde(values, bw_method=bandwidth)
        Z = np.reshape(kernel(positions).T, X.shape)
        print(f"Success ({func_name}): KDE calculation complete.")

        return Z, X, Y

    except Exception as e:
        print(f"An unexpected error occurred in {func_name}: {e}", file=sys.stderr)
        return None


def find_kde_minima_2d(Z, X, Y, sigma=2, min_distance=5, threshold_rel=0.01):
    """
    Finds local minima in a 2D KDE probability density grid.

    Uses Gaussian smoothing and peak finding on the *inverted* density.
    Requires scipy.ndimage and scikit-image.

    Args:
        Z (numpy.ndarray): The 2D KDE density grid (output from calculate_kde_2d).
        X (numpy.ndarray): The X coordinates of the grid mesh.
        Y (numpy.ndarray): The Y coordinates of the grid mesh.
        sigma (float, optional): Standard deviation for Gaussian filter smoothing.
                                 Increase for smoother landscapes, reducing noise peaks. Defaults to 2.
        min_distance (int, optional): Minimum number of pixels separating peaks found by
                                      peak_local_max. Determines how close minima can be. Defaults to 5.
        threshold_rel (float, optional): Minimum intensity relative to the maximum intensity
                                         (of the inverted density) for a peak (minimum) to be identified.
                                         Helps filter shallow noise minima. Defaults to 0.01 (1%).

    Returns:
        list or None: A list of tuples, where each tuple contains the (x, y) coordinates
                      of a found local minimum. Returns empty list if no minima are found.
                      Returns None if scikit-image/scipy.ndimage is unavailable or an error occurs.
    """
    func_name = "find_kde_minima_2d"
    if not _SKIMAGE_AVAILABLE or not _SCIPY_AVAILABLE:
        print(f"Error in {func_name}: SciPy (ndimage) and scikit-image are required for finding KDE minima.", file=sys.stderr)
        return None

    try:
        print(f"Info ({func_name}): Finding local minima in KDE grid (sigma={sigma}, min_dist={min_distance}, thresh={threshold_rel})...")
        # Smooth the density grid using gaussian_filter
        Z_smooth = gaussian_filter(Z, sigma=sigma)

        # Invert Z_smooth to find minima as local maxima
        Z_inv = -Z_smooth

        # Find local maxima in the inverted, smoothed density grid
        # peak_local_max returns coordinates in (row, column) format of the Z grid
        coordinates = peak_local_max(Z_inv,
                                     min_distance=min_distance,
                                     threshold_rel=threshold_rel) # Filter peaks below threshold

        if coordinates.size == 0:
             print(f"Warning in {func_name}: No local minima found with current parameters.")
             return [] # Return empty list if no minima found

        # Convert pixel coordinates (row, col) back to original data scale (x, y values)
        minima_coords = []
        for coord in coordinates:
            row_idx, col_idx = coord
            # X grid values correspond to columns, Y grid values correspond to rows
            x_coord = X[0, col_idx] # Get x value from the grid column
            y_coord = Y[row_idx, 0] # Get y value from the grid row
            minima_coords.append((x_coord, y_coord))

        print(f"Success ({func_name}): Found {len(minima_coords)} local minima.")
        return minima_coords

    except Exception as e:
        print(f"An unexpected error occurred in {func_name}: {e}", file=sys.stderr)
        return None


def calculate_distances(universe, selection_pairs, start_frame=0, stop_frame=-1, step=1):
    """
    Calculates distances between pairs of atom groups for each frame.

    Can calculate distance between the centers of mass (COM) of two groups
    or between two single atoms.

    Args:
        universe (MDAnalysis.Universe): An MDAnalysis Universe object.
        selection_pairs (list): A list of tuples. Each tuple should contain two
                                MDAnalysis selection strings. E.g.,
                                [("resid 10 and name CA", "resid 50 and name CA"),
                                 ("protein and name CA", "resname LIG")]
        start_frame (int, optional): First frame index to analyze. Defaults to 0.
        stop_frame (int, optional): Last frame index to analyze (exclusive).
                                    Defaults to -1 (until the end).
        step (int, optional): Step size for trajectory frames. Defaults to 1.

    Returns:
        pandas.DataFrame or None: A DataFrame where each column represents the
                                  distance for a selection pair over time.
                                  Columns are named 'Dist_Pair_X' where X is the index.
                                  An additional 'Frame' column indicates the frame index.
                                  Returns None if an error occurs or no pairs are provided.
    """
    func_name = "calculate_distances"
    if not selection_pairs:
        print(f"Warning in {func_name}: No selection pairs provided.", file=sys.stderr)
        return None

    try:
        n_pairs = len(selection_pairs)
        n_frames_total = universe.trajectory.n_frames
        # Use trajectory slicing to get exact frame indices
        traj_slice = universe.trajectory.slice[start_frame:stop_frame:step]
        frame_indices = traj_slice.indices
        n_frames_analyzed = len(frame_indices)

        if n_frames_analyzed == 0:
            print(f"Warning in {func_name}: No frames selected for analysis.", file=sys.stderr)
            return None

        print(f"Info ({func_name}): Calculating distances for {n_pairs} pairs over {n_frames_analyzed} frames...")

        # Prepare data storage
        distances_data = np.zeros((n_frames_analyzed, n_pairs))
        column_names = []

        # Create AtomGroup objects once if possible
        atom_groups1 = []
        atom_groups2 = []
        is_single_atom1 = []
        is_single_atom2 = []
        for i, (sel1_str, sel2_str) in enumerate(selection_pairs):
             try:
                  ag1 = universe.select_atoms(sel1_str)
                  ag2 = universe.select_atoms(sel2_str)
                  if len(ag1) == 0 or len(ag2) == 0:
                       raise ValueError(f"Selection resulted in 0 atoms for pair {i}: ('{sel1_str}', '{sel2_str}')")
                  atom_groups1.append(ag1)
                  atom_groups2.append(ag2)
                  is_single_atom1.append(len(ag1) == 1)
                  is_single_atom2.append(len(ag2) == 1)
                  # Create a more descriptive column name if possible
                  name1 = sel1_str.replace(" ", "_")[:15] # Truncate long names
                  name2 = sel2_str.replace(" ", "_")[:15]
                  column_names.append(f'Dist_{name1}__{name2}') # Double underscore separator
             except Exception as e:
                  print(f"Error creating atom groups for pair {i} ('{sel1_str}', '{sel2_str}'): {e}", file=sys.stderr)
                  return None

        # Iterate through trajectory frames using the slice
        for frame_idx_rel, ts in enumerate(traj_slice):
            for pair_idx in range(n_pairs):
                ag1 = atom_groups1[pair_idx]
                ag2 = atom_groups2[pair_idx]

                # Get positions based on whether they are single atoms or groups
                pos1 = ag1.positions[0] if is_single_atom1[pair_idx] else ag1.center_of_mass()
                pos2 = ag2.positions[0] if is_single_atom2[pair_idx] else ag2.center_of_mass()

                distance = np.linalg.norm(pos1 - pos2)
                distances_data[frame_idx_rel, pair_idx] = distance

            if frame_idx_rel % 100 == 0 or frame_idx_rel == n_frames_analyzed - 1: # Print progress
                 print(f"  Processed frame {ts.frame} ({frame_idx_rel+1}/{n_frames_analyzed})...", end='\r')

        print(f"\nSuccess ({func_name}): Distance calculations complete.")

        # Create DataFrame
        df_distances = pd.DataFrame(distances_data, columns=column_names)
        df_distances.insert(0, 'Frame', frame_indices)

        return df_distances

    except Exception as e:
         print(f"An unexpected error occurred in {func_name}: {e}", file=sys.stderr)
         return None


def calculate_rmsf(universe, selection="name CA", align=True, align_selection=None,
                   start_frame=0, stop_frame=-1, step=1,
                   in_memory_align=False):
    """
    Calculates the Root Mean Square Fluctuation (RMSF) for selected atoms.

    Optionally aligns the trajectory first. Uses MDAnalysis.analysis.rms.RMSF.

    Args:
        universe (MDAnalysis.Universe): An MDAnalysis Universe object.
        selection (str, optional): Atom selection string for RMSF calculation
                                   (often 'name CA' or 'protein'). Defaults to "name CA".
        align (bool, optional): Whether to perform RMSD alignment to the reference frame
                                (first frame of the slice) before RMSF calculation. Defaults to True.
        align_selection (str, optional): Atom selection string for alignment.
                                         If None and align is True, defaults to `selection`.
                                         Defaults to None.
        start_frame (int, optional): First frame index to analyze. Defaults to 0.
        stop_frame (int, optional): Last frame index to analyze (exclusive).
                                    Defaults to -1 (until the end).
        step (int, optional): Step size for trajectory frames. Defaults to 1.
        in_memory_align (bool, optional): Load trajectory into memory for alignment.
                                          Defaults to False.

    Returns:
        pandas.DataFrame or None: A DataFrame with columns like 'Atom_Index', 'Resid',
                                  'Resnum', 'Name', 'RMSF'. Returns None if an error
                                  occurs or no atoms are selected.
    """
    func_name = "calculate_rmsf"
    aligned_universe = universe # Start with original

    try:
        # Select atoms for RMSF calculation first to get indices etc.
        atoms_to_analyze = universe.select_atoms(selection)
        if not atoms_to_analyze:
             print(f"Warning in {func_name}: Selection '{selection}' resulted in 0 atoms.", file=sys.stderr)
             return None
        print(f"Info ({func_name}): Selected {len(atoms_to_analyze)} atoms for RMSF calculation.")

        # --- Alignment Step (Optional) ---
        if align:
            if align_selection is None:
                align_select = selection
            else:
                align_select = align_selection

            if not universe.select_atoms(align_select):
                 raise ValueError(f"Alignment selection '{align_select}' resulted in 0 atoms.")

            print(f"Info ({func_name}): Performing alignment using selection: '{align_select}'")
            try:
                # Align to the first frame of the specified range
                ref_actual_frame_idx = start_frame if start_frame is not None else 0
                ref_universe = universe # Use self as reference
                ref_universe.trajectory[ref_actual_frame_idx] # Go to reference frame

                aligner = align.AlignTraj(universe, ref_universe,
                                          select=align_select,
                                          in_memory=in_memory_align,
                                          filename=None) # Align in memory
                aligner.run(start=start_frame, stop=stop_frame, step=step)
                # Universe modified in place
                print(f"Info ({func_name}): Alignment completed.")
            except Exception as e:
                 print(f"Error in {func_name}: Alignment failed. {e}", file=sys.stderr)
                 return None
        else:
             print(f"Info ({func_name}): Skipping alignment.")

        # --- RMSF Calculation ---
        print(f"Info ({func_name}): Calculating RMSF...")
        # Important: Re-select atoms from the *aligned* universe for RMSF calculation
        atoms_in_aligned = aligned_universe.select_atoms(selection)
        if not atoms_in_aligned:
             print(f"Error in {func_name}: Lost atoms after alignment step? Check selections.", file=sys.stderr)
             return None

        rmsf_analysis = rms.RMSF(atoms_in_aligned, verbose=True) # Add verbose=True for progress
        # Run on the specified frame range of the (potentially aligned) universe
        rmsf_analysis.run(start=start_frame, stop=stop_frame, step=step)

        rmsf_values = rmsf_analysis.rmsf

        print(f"Success ({func_name}): RMSF calculation complete for {len(rmsf_values)} atoms.")

        # Create a DataFrame for output including atom info
        df_rmsf = pd.DataFrame({
             'Atom_Index': atoms_in_aligned.indices,
             'Resid': atoms_in_aligned.resids,
             'Resnum': atoms_in_aligned.resnums,
             'Name': atoms_in_aligned.names,
             'RMSF': rmsf_values
        })

        return df_rmsf

    except Exception as e:
         print(f"An unexpected error occurred in {func_name}: {e}", file=sys.stderr)
         return None


def calculate_radius_of_gyration(universe, selection="protein and name CA",
                                 start_frame=0, stop_frame=-1, step=1):
    """
    Calculates the Radius of Gyration (Rg) for a selection over a trajectory.

    Args:
        universe (MDAnalysis.Universe): An MDAnalysis Universe object.
        selection (str, optional): Atom selection string for Rg calculation.
                                   Defaults to "protein and name CA".
        start_frame (int, optional): First frame index to analyze. Defaults to 0.
        stop_frame (int, optional): Last frame index to analyze (exclusive).
                                    Defaults to -1 (until the end).
        step (int, optional): Step size for trajectory frames. Defaults to 1.

    Returns:
        pandas.DataFrame or None: A DataFrame with 'Frame' and 'Rg' columns,
                                  containing the radius of gyration for each
                                  analyzed frame. Returns None if an error occurs
                                  or no atoms are selected.
    """
    func_name = "calculate_radius_of_gyration"
    try:
        atom_group = universe.select_atoms(selection)
        if not atom_group:
             print(f"Warning in {func_name}: Selection '{selection}' resulted in 0 atoms.", file=sys.stderr)
             return None

        # Use trajectory slicing to get exact frame indices
        traj_slice = universe.trajectory.slice[start_frame:stop_frame:step]
        frame_indices = traj_slice.indices
        n_frames_analyzed = len(frame_indices)

        if n_frames_analyzed == 0:
            print(f"Warning in {func_name}: No frames selected for analysis.", file=sys.stderr)
            return None

        print(f"Info ({func_name}): Calculating Radius of Gyration for selection '{selection}' over {n_frames_analyzed} frames...")

        rg_values = np.zeros(n_frames_analyzed)

        # Iterate using the trajectory slice
        for frame_idx_rel, ts in enumerate(traj_slice):
            # Rg is calculated on the current frame's coordinates for the atom_group
            rg = atom_group.radius_of_gyration()
            rg_values[frame_idx_rel] = rg

            if frame_idx_rel % 100 == 0 or frame_idx_rel == n_frames_analyzed - 1: # Print progress
                 print(f"  Processed frame {ts.frame} ({frame_idx_rel+1}/{n_frames_analyzed})...", end='\r')

        print(f"\nSuccess ({func_name}): Radius of Gyration calculation complete.")

        # Create DataFrame
        df_rg = pd.DataFrame({'Frame': frame_indices, 'Rg': rg_values})

        return df_rg

    except Exception as e:
         print(f"An unexpected error occurred in {func_name}: {e}", file=sys.stderr)
         return None


def calculate_rmsd_relative_to_ref(universe, reference_universe,
                                   selection="name CA", align_selection=None,
                                   start_frame=0, stop_frame=-1, step=1):
    """
    Calculates the Root Mean Square Deviation (RMSD) between trajectory frames
    and a reference structure.

    Alignment (superposition) is performed before RMSD calculation based on
    `align_selection`.

    Args:
        universe (MDAnalysis.Universe): Universe object for the trajectory.
        reference_universe (MDAnalysis.Universe): Universe object for the reference
                                                 structure (should contain at least one frame).
        selection (str, optional): Atom selection string for calculating RMSD
                                   (must be valid and have same atom order in both
                                   universe and reference). Defaults to "name CA".
        align_selection (str, optional): Atom selection string for superposition before
                                         RMSD calculation (must be valid and have same
                                         atom order in both). If None, defaults to `selection`.
                                         Defaults to None.
        start_frame (int, optional): First frame index of the trajectory to analyze. Defaults to 0.
        stop_frame (int, optional): Last frame index of the trajectory to analyze (exclusive).
                                    Defaults to -1 (until the end).
        step (int, optional): Step size for trajectory frames. Defaults to 1.

    Returns:
        pandas.DataFrame or None: A DataFrame with 'Frame' and 'RMSD' columns,
                                  containing the RMSD value for each analyzed frame
                                  relative to the reference. Returns None if an error occurs.
    """
    func_name = "calculate_rmsd_relative_to_ref"
    try:
        if align_selection is None:
            align_selection = selection # Default to using RMSD selection for alignment

        # Select atoms from trajectory universe (will be updated each frame)
        traj_atoms_rmsd = universe.select_atoms(selection)
        traj_atoms_align = universe.select_atoms(align_selection)

        # Select atoms from reference universe (from its first frame)
        reference_universe.trajectory[0] # Ensure we are on the reference frame
        ref_atoms_rmsd = reference_universe.select_atoms(selection)
        ref_atoms_align = reference_universe.select_atoms(align_selection)
        # Store reference coordinates
        ref_pos_rmsd = ref_atoms_rmsd.positions.copy()
        ref_pos_align = ref_atoms_align.positions.copy()


        # --- Sanity Checks ---
        if not traj_atoms_rmsd or not ref_atoms_rmsd:
             raise ValueError(f"RMSD selection '{selection}' resulted in 0 atoms in trajectory or reference.")
        if not traj_atoms_align or not ref_atoms_align:
             raise ValueError(f"Alignment selection '{align_selection}' resulted in 0 atoms in trajectory or reference.")
        if len(traj_atoms_rmsd) != len(ref_atoms_rmsd):
             raise ValueError(f"RMSD selections have different numbers of atoms: Traj={len(traj_atoms_rmsd)}, Ref={len(ref_atoms_rmsd)}")
        if len(traj_atoms_align) != len(ref_atoms_align):
             raise ValueError(f"Alignment selections have different numbers of atoms: Traj={len(traj_atoms_align)}, Ref={len(ref_atoms_align)}")

        print(f"Info ({func_name}): Calculating RMSD using selection '{selection}' relative to reference.")
        print(f"Info ({func_name}): Aligning using selection '{align_selection}'.")

        # Use trajectory slicing to get exact frame indices
        traj_slice = universe.trajectory.slice[start_frame:stop_frame:step]
        frame_indices = traj_slice.indices
        n_frames_analyzed = len(frame_indices)

        if n_frames_analyzed == 0:
            print(f"Warning in {func_name}: No frames selected for analysis.", file=sys.stderr)
            return None

        print(f"Info ({func_name}): Analyzing {n_frames_analyzed} frames...")

        rmsd_values = np.zeros(n_frames_analyzed)

        # Iterate through trajectory frames using the slice
        for frame_idx_rel, ts in enumerate(traj_slice):
            # Get current coordinates for alignment and RMSD calculation selections
            traj_pos_align = traj_atoms_align.positions
            traj_pos_rmsd = traj_atoms_rmsd.positions

            # Calculate RMSD using the rms.rmsd function
            # This performs alignment based on align selections, then calculates RMSD
            # based on the rmsd selections using the derived rotation matrix.
            rmsd_val = rms.rmsd(
                traj_pos_rmsd,  # mobile coordinates for RMSD
                ref_pos_rmsd,   # reference coordinates for RMSD
                mobile_atoms=traj_pos_align, # Coordinates used FOR alignment
                ref_atoms=ref_pos_align,   # Reference coordinates used FOR alignment
                superposition=True        # Perform alignment
                # weights='mass' # Optional mass weighting
            )

            rmsd_values[frame_idx_rel] = rmsd_val

            if frame_idx_rel % 100 == 0 or frame_idx_rel == n_frames_analyzed - 1: # Print progress
                 print(f"  Processed frame {ts.frame} ({frame_idx_rel+1}/{n_frames_analyzed}), RMSD = {rmsd_val:.3f} Å...", end='\r')

        print(f"\nSuccess ({func_name}): RMSD calculation complete.")

        # Create DataFrame
        df_rmsd = pd.DataFrame({'Frame': frame_indices, 'RMSD': rmsd_values})

        return df_rmsd

    except ValueError as e: # Catch selection errors
         print(f"Error in {func_name}: {e}", file=sys.stderr)
         return None
    except Exception as e:
         print(f"An unexpected error occurred in {func_name}: {e}", file=sys.stderr)
         return None


# End of md_analysis_tools.py
