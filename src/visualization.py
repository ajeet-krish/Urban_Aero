import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from phi import math
from phi.field import CenteredGrid, StaggeredGrid

def get_2d_slice(field, z_idx):
    # If field has a vector dimension, take the norm to make it a scalar field
    # But for streamplot, we need the raw components. Let's handle cases.
    if 'vector' in field.shape:
        # Return components or magnitude depending on need
        pass 
        
    slice_2d = field.z[z_idx]
    return slice_2d.values.numpy('x,y')

def get_2d_vector_slice(field, z_idx):
    # Expects a vector field, returns u and v components
    slice_2d = field.z[z_idx]
    # Assuming centered grid with vector dimension
    data = slice_2d.values.numpy('x,y,vector')
    return data[..., 0], data[..., 1] # u, v

def create_animation(frames, obstacle_mask, filename="assets/simulation.gif", fps=10, z_slice=2):
    """
    frames: list of tuples (velocity_centered, vort_grid, smoke_grid)
    """
    import os
    if not os.path.exists("assets"):
        os.makedirs("assets")

    mask_data = get_2d_slice(obstacle_mask, z_slice)
    
    # Grid for streamlines
    res_x, res_y = mask_data.shape
    X, Y = np.meshgrid(np.linspace(0, 200, res_x), np.linspace(0, 200, res_y), indexing='ij')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Initialize with first frame
    vel_centered, vort_grid, smoke_grid = frames[0]
    u, v = get_2d_vector_slice(vel_centered, z_slice)
    vort_slice = get_2d_slice(vort_grid, z_slice)
    smoke_slice = get_2d_slice(smoke_grid, z_slice)
    
    # Wind Speed (with streamlines)
    im1 = axes[0].imshow(np.linalg.norm([u, v], axis=0).T, origin='lower', cmap='viridis', extent=[0, 200, 0, 200], vmin=0, vmax=0.5)
    axes[0].streamplot(X.T, Y.T, u.T, v.T, color='white', density=0.5)
    axes[0].contour(mask_data.T, levels=[0.5], colors='white', extent=[0, 200, 0, 200])
    axes[0].set_title("Wind Speed [m/s]")
    plt.colorbar(im1, ax=axes[0])
    
    # Vorticity (with streamlines)
    im2 = axes[1].imshow(vort_slice.T, origin='lower', cmap='RdBu_r', extent=[0, 200, 0, 200], vmin=-0.1, vmax=0.1)
    axes[1].streamplot(X.T, Y.T, u.T, v.T, color='black', density=0.5)
    axes[1].contour(mask_data.T, levels=[0.5], colors='black', extent=[0, 200, 0, 200])
    axes[1].set_title("Vorticity [1/s]")
    plt.colorbar(im2, ax=axes[1])
    
    # Smoke Concentration
    im3 = axes[2].imshow(smoke_slice.T, origin='lower', cmap='inferno', extent=[0, 200, 0, 200], vmin=0, vmax=0.1)
    axes[2].contour(mask_data.T, levels=[0.5], colors='white', extent=[0, 200, 0, 200])
    axes[2].set_title("Smoke Concentration")
    plt.colorbar(im3, ax=axes[2])

    def update(frame_idx):
        vel_centered, vort_grid, smoke_grid = frames[frame_idx]
        u, v = get_2d_vector_slice(vel_centered, z_slice)
        im1.set_data(np.linalg.norm([u, v], axis=0).T)
        im2.set_data(get_2d_slice(vort_grid, z_slice).T)
        im3.set_data(get_2d_slice(smoke_grid, z_slice).T)
        return im1, im2, im3

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=True)
    ani.save(filename, writer='pillow')
    print(f"Animation saved to {filename}")

