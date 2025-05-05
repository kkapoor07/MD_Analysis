# Python MD Analysis Tools

## Overview

This repository provides a Python module (`md_analysis_tools.py`) containing a collection of functions designed for analyzing molecular dynamics (MD) trajectories. It leverages libraries like [MDAnalysis](https://www.mdanalysis.org/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Scikit-learn](https://scikit-learn.org/), and [SciPy](https://scipy.org/) to perform common analysis tasks often encountered in computational chemistry and biophysics research.

The goal is to offer reusable tools for calculating metrics, performing dimensionality reduction, clustering conformations, and analyzing conformational landscapes. Example Jupyter notebooks are included to demonstrate typical usage workflows.

## Key Features

The core module `md_analysis_tools.py` includes functions for:

*   **Geometric Analysis:**
    *   Calculating Phi/Psi dihedral angles (`calculate_phi_psi_angles`)
    *   Calculating distances between atom groups/atoms (`calculate_distances`)
    *   Calculating Radius of Gyration (`calculate_radius_of_gyration`)
*   **Fluctuations & Comparisons:**
    *   Calculating Root Mean Square Fluctuation (RMSF) per atom (`calculate_rmsf`)
    *   Calculating Root Mean Square Deviation (RMSD) relative to a reference structure (`calculate_rmsd_relative_to_ref`)
*   **Dimensionality Reduction:**
    *   Performing Cartesian Principal Component Analysis (PCA) (`perform_cartesian_pca`)
*   **Clustering & State Identification:**
    *   Performing KMeans clustering (`perform_kmeans`)
    *   Finding representative trajectory frames closest to cluster centroids (`find_closest_frames_to_centroids`)
    *   Calculating 2D Kernel Density Estimates (KDE) for landscape analysis (`calculate_kde_2d`)
    *   Finding local minima in 2D KDE landscapes (`find_kde_minima_2d` - requires `scikit-image`)

## Installation / Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/kkapoor07/Python_MD_Analysis.git
    cd Python_MD_Analysis
    ```
    *(Replace `your-github-username`)*

2.  **Install Dependencies:** It is highly recommended to use a virtual environment (like `conda` or `venv`).
    ```bash
    # Using conda (recommended)
    # conda create -n mdanalysis python=3.9 # Or your preferred Python version
    # conda activate mdanalysis
    # conda install --file requirements.txt -c conda-forge # Installs most deps from conda-forge

    # --- OR ---

    # Using pip and venv
    # python -m venv env
    # source env/bin/activate # On Linux/macOS
    # # .\env\Scripts\activate # On Windows
    # pip install -r requirements.txt
    ```
    See the `requirements.txt` file for the list of necessary packages. Note that `scikit-image` is optional but required for the `find_kde_minima_2d` function.

## Usage

The primary way to use this toolkit is to import the functions from `md_analysis_tools.py` into your own Python scripts or Jupyter notebooks.

```python
import md_analysis_tools
import MDAnalysis as mda

# Load your trajectory
u = mda.Universe("your_topology.prmtop", "your_trajectory.dcd")
u_ref = mda.Universe("your_reference.pdb") # If needed for RMSD

# Example: Calculate RMSD
df_rmsd = md_analysis_tools.calculate_rmsd_relative_to_ref(
    universe=u,
    reference_universe=u_ref,
    selection="protein and name CA"
)
if df_rmsd is not None:
    print(df_rmsd.head())
    # Add plotting code here...

# Example: Perform PCA
pca_result = md_analysis_tools.perform_cartesian_pca(
    universe=u,
    select="protein and name CA and resid 10:100",
    n_components=5
)
if pca_result:
     print("PCA Variance:", pca_result.variance)
     # Project trajectory, plot, etc...
```

Please refer to the example Jupyter notebooks included in this repository for detailed workflows demonstrating how to use each function and visualize the results:

*   **`Example_Basic_Metrics.ipynb`**: Shows usage of distance, RMSF, Rg, and RMSD calculations.
*   **`Example_PCA_Analysis.ipynb`**: Demonstrates Cartesian PCA, variance analysis, and projection plotting.
*   **`Example_Clustering_Analysis.ipynb`**: Shows KMeans clustering (typically on PCA data), elbow method, visualization, and finding representative structures.
*   **`Example_Landscape_Analysis.ipynb`**: Covers 2D KDE calculation, minima finding (state identification), visualization, and extracting representatives from landscape minima.

## Dependencies

See `requirements.txt`. Core dependencies include:

*   Python (>= 3.7 recommended)
*   MDAnalysis (>= 2.0.0 recommended)
*   NumPy
*   Pandas
*   Scikit-learn (for KMeans)
*   SciPy (for KDE, distance calculations)
*   Matplotlib / Seaborn (for plotting in examples)
*   Scikit-image (Optional, for KDE minima finding)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This code is provided as-is. While developed with care, it's primarily intended as a demonstration toolkit. Robust error handling for all edge cases in trajectory/topology formats or analysis parameters may not be exhaustive. Users should validate results for their specific systems. For highly complex or non-standard PDB/trajectory files, using the underlying libraries (MDAnalysis, etc.) directly or exploring more specialized tools might be necessary.
