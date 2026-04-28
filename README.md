# Urban Wind Canyons

A differentiable 3D CFD analysis framework for pedestrian comfort in urban environments.

## Overview

Modern urban planning often inadvertently creates "wind canyons", areas where building geometry accelerates wind, leading to dangerous gusts at the pedestrian level. This project utilizes **Differentiable Physics** to simulate 3D wind flow and pollutant dispersion.

The framework is built on [PhiFlow](https://github.com/tum-pbs/PhiFlow) and leverages GPU acceleration via PyTorch, making it suitable for both engineering analysis and high-end cinematic visualization.

## Features

*   **Differentiable Physics:** Enables optimization of building geometry for pedestrian comfort.
*   **3D CFD Engine:** Solves the 3D Incompressible Navier-Stokes equations and Advection-Diffusion for pollutants.
*   **Procedural Urban Modeling:** Programmatic generation of city blocks for rapid iteration.
*   **Full Pipeline:** From physical simulation to data export (`.npz`) compatible with Blender Geometry Nodes.
*   **Accelerated:** Optimized for Apple Silicon (MPS backend) using PyTorch.

## Mathematical Formulation

The system solves the 3D Incompressible Newtonian fluid equations:

1.  **Momentum:** $\rho \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} \right) = -\nabla p + \mu \nabla^2 \mathbf{u} + \mathbf{f}$
2.  **Continuity:** $\nabla \cdot u = 0$

Pollutant (e.g., $CO_2$) transport is modeled with the advection-diffusion equation:
$$\frac{\partial c}{\partial t} + \mathbf{u} \cdot \nabla c = D \nabla^2 c + S$$

## Getting Started

### Prerequisites

*   `uv` (Package manager)
*   Python 3.14+

### Installation

```bash
uv sync
```

### Running the Simulation

```bash
uv run run_sim.py
```

This will run a 50-step simulation, outputting the wind/smoke field results in the `data/` directory.

## Documentation

This project uses [Quarto](https://quarto.org/) for literate programming and documentation. See `index.qmd` for the technical theory and interactive results analysis.
