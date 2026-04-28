import numpy as np
import os
from phi.torch.flow import *

def export_fields(velocity, smoke, step, output_dir="data"):
    """
    Exports velocity and smoke fields to .npz files for Blender/post-processing.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Convert fields to centered grids for easy exporting
    vel_centered = velocity.at_centers()
    
    # Get numpy arrays (ensure spatial ordering)
    # unstack() returns a tuple of Tensors, we need to convert each component
    vel_components = vel_centered.vector.unstack()
    vel_data = np.stack([c.numpy('x,y,z') for c in vel_components], axis=0)
    
    smoke_data = smoke.values.numpy('x,y,z')
    
    filename = os.path.join(output_dir, f"sim_step_{step:04d}.npz")
    np.savez_compressed(filename, velocity=vel_data, smoke=smoke_data)
    print(f"Exported data to {filename}")

