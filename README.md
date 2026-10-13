# Bachelor's Thesis: Magnetohydrostatic Modeling & Wilson Depression in Sunspots

This repository contains the complete computational framework developed for my Bachelor's thesis in Mathematics at Albert-Ludwigs-Universität Freiburg, conducted in collaboration with the **Kiepenheuer Institute for Solar Physics (KIS)**. 

The project focuses on solving magnetohydrostatic (MHS) equilibrium equations using finite difference schemes to model solar atmospheres (Quiet Sun vs. Sunspots/Umbra) and quantify the **Wilson Depression**.

---

## Key Features & Architecture

The codebase is structured into modular components separating core numerical algorithms, analytical models, and 3D MHD data processing pipelines:

1. **`numerical_tools.py` (Core Numerics):**
   * **Gaussian Elimination (`gaus`):** Robust custom linear system solver with partial pivoting.
   * **Finite Difference Schemes (FDM):** High-order (up to 2nd order) derivative approximations for uniform and **variable spatial grids** ($\Delta z$).
   * **Custom Boundary Stencils (`Stencil_*`):** Matrix builders handling Dirichlet, Neumann, and mixed boundary conditions, including **Ghost Cell** implementations for precise boundary flux handling.
   * **Poisson Solvers:** Specialized solvers for boundary value problems in stratified atmospheres.
   * **Numerical Integration:** Cumulative trapezoidal integration routines for optical depth ($\tau$) profiling.

2. **`analytical_pressure_solver.py` & `MHD_4.py` (Astrophysical Pipelines):**
   * **3D MHD Data Parsing:** Efficient handling of raw Fortran binary files (`FortranFile`) from 3D solar simulation grids (`[Z, Y, X]`).
   * **Magnetic Pressure Integration:** Incorporating magnetic field vectors ($B_x, B_y, B_z$) to compute magnetic pressure ($P_{mag} = B^2 / 8\pi$) and modify hydrostatic equilibrium.
   * **Optical Depth & Wilson Depression:** Automated routines to locate optical surfaces ($\tau = 1$) via interpolation and compute the geometrical height difference (Wilson Depression) between the Quiet Sun and sunspot umbrae.
   * **Diagnostic Visualizations:** Automated multi-panel publication-grade plotting (logarithmic pressure stratification, temperature profiles, and 2D horizontal X-Y cross-sections exported as vector PDFs).

---

## Repository Structure

```text
├── numerical_tools.py                 # Custom FDM stencils, matrix solvers, and integration tools
├── analytical_pressure_solver.py      # Analytical/dynamic temperature profiling & 1D MHS solver
├── MHD_4.py                           # 3D MHD binary data processing and spatial analysis
└── README.md                          # Project documentation# Bachelor's Thesis: Magnetohydrostatic Modeling & Wilson Depression in Sunspots

This repository contains the complete computational framework developed for my Bachelor's thesis in Mathematics at Albert-Ludwigs-Universität Freiburg, conducted in collaboration with the **Kiepenheuer Institute for Solar Physics (KIS)**. 

The project focuses on solving magnetohydrostatic (MHS) equilibrium equations using finite difference schemes to model solar atmospheres (Quiet Sun vs. Sunspots/Umbra) and quantify the **Wilson Depression**.

---

## Key Features & Architecture

The codebase is structured into modular components separating core numerical algorithms, analytical models, and 3D MHD data processing pipelines:

1. **`numerical_tools.py` (Core Numerics):**
   * **Gaussian Elimination (`gaus`):** Robust custom linear system solver with partial pivoting.
   * **Finite Difference Schemes (FDM):** High-order (up to 2nd order) derivative approximations for uniform and **variable spatial grids** (`$\Delta z$`).
   * **Custom Boundary Stencils (`Stencil_*`):** Matrix builders handling Dirichlet, Neumann, and mixed boundary conditions, including **Ghost Cell** implementations for precise boundary flux handling.
   * **Poisson Solvers:** Specialized solvers for boundary value problems in stratified atmospheres.
   * **Numerical Integration:** Cumulative trapezoidal integration routines for optical depth ($\tau$) profiling.

2. **`Berechnung_25_6.py` & `MHD_4.py` (Astrophysical Pipelines):**
   * **3D MHD Data Parsing:** Efficient handling of raw Fortran binary files (`FortranFile`) from 3D solar simulation grids (`[Z, Y, X]`).
   * **Magnetic Pressure Integration:** Incorporating magnetic field vectors ($B_x, B_y, B_z$) to compute magnetic pressure ($P_{mag} = B^2 / 8\pi$) and modify hydrostatic equilibrium.
   * **Optical Depth & Wilson Depression:** Automated routines to locate optical surfaces ($\tau = 1$) via interpolation and compute the geometrical height difference (Wilson Depression) between the Quiet Sun and sunspot umbrae.
   * **Diagnostic Visualizations:** Automated multi-panel publication-grade plotting (logarithmic pressure stratification, temperature profiles, and 2D horizontal X-Y cross-sections exported as vector PDFs).

---

## Repository Structure

```text
├── numerical_tools.py       # Custom FDM stencils, matrix solvers, and integration tools
├── Berechnung_25_6.py       # Analytical/dynamic temperature profiling & 1D MHS solver
├── MHD_4.py                 # 3D MHD binary data processing and spatial analysis
└── README.md                # Project documentation

---

## Requirements

The codebase requires a standard scientific Python stack:
* **Python 3.x**
* **NumPy**
* **SciPy** (specifically `scipy.integrate` and `scipy.io.FortranFile`)
* **Matplotlib**

---

## Acknowledgments

* **Continuum Opacity Calculations (`cont_opac`):** The routines for calculating continuous opacities were provided by [Andreu](https://andreuva.com/cv.html) as part of the research collaboration at the Kiepenheuer Institute for Solar Physics (KIS). All custom numerical solvers, FDM stencils, and MHD data processing pipelines were developed independently.

---

## Author

**Jannik Alexander Mittnacht**  
[GitHub Profile](https://github.com/JannikMittnacht)
