#%% md
# # Example: Cartesian Principal Component Analysis (PCA) of MD Trajectory

# This notebook demonstrates how to use the `perform_cartesian_pca` function from the `md_analysis_tools` library to perform PCA on protein C-alpha atoms from a molecular dynamics trajectory.

# **Workflow:**
# 1. Import necessary libraries.
# 2. Load the simulation topology and trajectory using MDAnalysis.
# 3. Define atom selections for alignment and PCA.
# 4. Run the PCA calculation using `perform_cartesian_pca`.
# 5. Analyze the results (explained variance).
# 6. Visualize the trajectory projected onto the principal components.
# 7. (Optional) Generate structures/trajectories representing motion along principal components.

#%%
# Import necessary libraries
import md_analysis_tools # Our custom library
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import os

# Configure plotting style (optional)
plt.style.use('seaborn-v0_8-poster')

#%% md
# ## 1. Load Simulation Data

# We need to load the simulation topology (e.g., PRMTOP, PSF, PDB) and trajectory (e.g., DCD, XTC, NC) files into an MDAnalysis Universe object.

# **ACTION:** Replace the placeholder file paths below with the actual paths to your topology and trajectory files.

#%%
# --- User Input: Define File Paths ---
topology_file = "placeholder.prmtop" # <-- REPLACE with your topology file (e.g., protein.prmtop)
trajectory_file = "placeholder.dcd"   # <-- REPLACE with your trajectory file (e.g., trajectory.dcd)
output_dir = "pca_analysis_output"    # Optional: Directory to save results

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# --- Load Universe ---
print(f"Loading trajectory...")
try:
    # Check if files exist before loading
    if not os.path.exists(topology_file) or not os.path.exists(trajectory_file):
         raise FileNotFoundError(f"Ensure topology ('{topology_file}') and trajectory ('{trajectory_file}') files exist.")

    # Load the universe
    u = mda.Universe(topology_file, trajectory_file)
    print(f"Successfully loaded Universe:")
    print(f"  Number of atoms: {len(u.atoms)}")
    print(f"  Number of frames: {len(u.trajectory)}")
except Exception as e:
    print(f"Error loading Universe: {e}", file=sys.stderr)
    # You might want to stop execution here if loading fails
    # exit() # Uncomment to stop if loading fails

#%% md
# ## 2. Define Atom Selections

# Specify the group of atoms to be used for the PCA calculation (typically C-alpha atoms of the protein) and, if alignment is desired, the atoms used for the alignment (often the same C-alpha atoms or a stable core).

#%%
# --- User Input: Define Selections ---

# Selection for PCA calculation (e.g., C-alpha atoms, excluding terminals if desired)
pca_selection = "name CA and protein"
# Example excluding terminals: "name CA and protein and resid 5:295"

# Selection for RMSD alignment (can be the same as pca_selection or a more stable region)
# Set to None to skip alignment step if trajectory is already aligned
align_selection = "name CA and protein"

# --- Optional: Check selections ---
try:
    pca_atoms = u.select_atoms(pca_selection)
    print(f"PCA selection ('{pca_selection}') includes {len(pca_atoms)} atoms.")
    if align_selection:
        align_atoms = u.select_atoms(align_selection)
        print(f"Alignment selection ('{align_selection}') includes {len(align_atoms)} atoms.")
    else:
        print("Alignment will be skipped.")
    if len(pca_atoms) == 0:
         raise ValueError("PCA selection resulted in 0 atoms. Check your selection string.")
except Exception as e:
    print(f"Error checking selections: {e}", file=sys.stderr)
    # exit() # Uncomment to stop if selections are invalid

#%% md
# ## 3. Perform PCA Calculation

# Now, we call the `perform_cartesian_pca` function from our library. We can specify the number of components (`n_components`) or leave it as `None` to calculate all. We also pass the alignment flag and selections.

#%%
# --- Run PCA ---
print("\nPerforming PCA...")

# Call the function from our library
# Set align=False if align_selection is None
pca_result = md_analysis_tools.perform_cartesian_pca(
    universe=u,
    select=pca_selection,
    align=(align_selection is not None), # Only align if align_selection is provided
    align_select=align_selection,
    n_components=10, # Calculate the top 10 components (adjust as needed, None for all)
    # Adjust frame range if needed (defaults to all frames)
    # start_frame=0,
    # stop_frame=-1,
    # step=1,
    in_memory_align=False # Use False for large trajectories to save memory
)

if pca_result is None:
    print("PCA calculation failed. Please check previous error messages.", file=sys.stderr)
    # exit() # Uncomment to stop if PCA fails
else:
    print("PCA object created successfully.")

#%% md
# ## 4. Analyze Explained Variance

# The `pca_result` object contains the variance explained by each principal component. We can plot the cumulative variance to see how many components are needed to capture a significant portion of the motion.

#%%
# --- Analyze Variance ---
if pca_result:
    n_pcs = pca_result.n_components
    explained_variance = pca_result.variance[:n_pcs]
    cumulative_variance = pca_result.cumulated_variance[:n_pcs]

    print(f"\nExplained variance per component:\n{explained_variance}")
    print(f"\nCumulative variance:\n{cumulative_variance}")

    # Plot cumulative variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_pcs + 1), cumulative_variance, marker='o', linestyle='--')
    plt.title('Cumulative Explained Variance by Principal Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.xticks(range(1, n_pcs + 1))
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_cumulative_variance.png"), dpi=300)
    plt.show()

    # Print variance covered by first few PCs
    print(f"\nVariance covered by first 3 PCs: {cumulative_variance[2]:.3f}")
else:
    print("Skipping variance analysis because PCA failed.")

#%% md
# ## 5. Visualize Projections

# We can project the trajectory onto the first few principal components (e.g., PC1 and PC2) to visualize the conformational space sampled during the simulation. We use the `transform` method of the `pca_result` object.

#%%
# --- Visualize Projections ---
if pca_result:
    # Select the atoms used for PCA again (needed for transform)
    pca_atoms = u.select_atoms(pca_selection)

    # Transform the trajectory onto the first 2 principal components
    n_transform_pcs = 2
    print(f"\nProjecting trajectory onto first {n_transform_pcs} PCs...")
    projected_trajectory = pca_result.transform(pca_atoms, n_components=n_transform_pcs)
    print(f"Shape of projected trajectory: {projected_trajectory.shape}") # (n_frames, n_transform_pcs)

    # Create a DataFrame for easier plotting
    df_projections = pd.DataFrame(projected_trajectory, columns=[f'PC{i+1}' for i in range(n_transform_pcs)])

    # Add a time column (assuming constant time step)
    # Get dt in ns if possible, otherwise assume frames
    try:
        dt_ns = u.trajectory.dt / 1000.0 # Assuming dt is in ps
        df_projections['Time (ns)'] = df_projections.index * dt_ns * (u.trajectory.skip_timestep if hasattr(u.trajectory, 'skip_timestep') else 1)
        time_label = 'Time (ns)'
    except:
        df_projections['Frame'] = df_projections.index
        time_label = 'Frame'

    print(df_projections.head())

    # Create scatter plot (PC1 vs PC2), colored by time
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(df_projections['PC1'], df_projections['PC2'], c=df_projections[time_label], cmap='viridis', s=10, alpha=0.6)
    plt.title('Trajectory Projection onto PC1 and PC2')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle=':')
    cbar = plt.colorbar(scatter)
    cbar.set_label(time_label)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_projection_pc1_pc2.png"), dpi=300)
    plt.show()

else:
    print("Skipping projection visualization because PCA failed.")

#%% md
# ## 6. (Optional) Visualize Motion Along PCs

# We can also generate a pseudo-trajectory that represents the motion along a specific principal component. This helps visualize the collective motion captured by that component.

#%%
# --- Visualize Motion (Optional) ---
if pca_result:
    pc_index_to_visualize = 0 # Visualize motion along PC1 (index 0)
    n_frames_viz = 20 # Number of frames for the pseudo-trajectory

    print(f"\nGenerating visualization for PC{pc_index_to_visualize + 1}...")

    # Select the atoms used for PCA
    pca_atoms = u.select_atoms(pca_selection)

    # Get the principal component vector and mean structure coordinates
    pc_vector = pca_result.p_components[:, pc_index_to_visualize]
    mean_coords_flat = pca_result.mean.flatten() # Ensure mean is flattened

    # Determine range of motion based on projection values
    transformed_pc = pca_result.transform(pca_atoms, n_components=pc_index_to_visualize + 1)
    pc_projections = transformed_pc[:, pc_index_to_visualize]
    min_proj, max_proj = np.min(pc_projections), np.max(pc_projections)

    # Create frames interpolating along the PC from min to max projection value
    motion_range = np.linspace(min_proj, max_proj, n_frames_viz)

    # Calculate coordinates for each frame: mean + (projection * PC_vector)
    # Reshape required: projection scalar * PC_vector (flat) -> add to mean (flat) -> reshape to (n_atoms, 3)
    vis_coordinates = np.array([
        (mean_coords_flat + scale * pc_vector).reshape(-1, 3) for scale in motion_range
    ])

    print(f"Shape of generated coordinates for visualization: {vis_coordinates.shape}") # Should be (n_frames_viz, n_pca_atoms, 3)

    # Create a new Universe or AtomGroup to hold these coordinates
    # Important: Use only the atoms included in the PCA for this visualization
    vis_atoms = pca_atoms.copy() # Create a copy to avoid modifying the original selection
    vis_universe = mda.Merge(vis_atoms)
    vis_universe.load_new(vis_coordinates, order="fac")

    # --- Save the visualization ---
    pdb_filename = os.path.join(output_dir, f"pca_motion_pc{pc_index_to_visualize + 1}.pdb")
    dcd_filename = os.path.join(output_dir, f"pca_motion_pc{pc_index_to_visualize + 1}.dcd")

    try:
        # Save as multi-model PDB
        vis_universe.atoms.write(pdb_filename, frames='all')
        print(f"Saved PDB visualization: {pdb_filename}")

        # Save as DCD trajectory
        vis_universe.atoms.write(dcd_filename, frames='all')
        print(f"Saved DCD visualization: {dcd_filename}")

        # You can now load these files into VMD, PyMOL, etc. to view the motion.
        # Example VMD command: vmd pca_motion_pc1.pdb -dcd pca_motion_pc1.dcd

    except Exception as e:
        print(f"Error saving visualization files: {e}", file=sys.stderr)

else:
    print("Skipping motion visualization because PCA failed.")


#%% md
# ## Conclusion

# This notebook demonstrated loading trajectory data, performing Cartesian PCA using the `md_analysis_tools` library, analyzing the explained variance, and visualizing the results through projections and motion along principal components.
