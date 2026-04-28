from phi.torch.flow import *
import torch
import phi.field
from phi.field import StaggeredGrid, CenteredGrid, curl

def setup_simulation(resolution=(64, 64, 16), domain_size=(200, 200, 50)):
    # Configure backend
    math.set_global_precision(32)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running on: {device}")
    
    # Domain
    bounds = Box(x=domain_size[0], y=domain_size[1], z=domain_size[2])
    
    # Velocity (Staggered for stability)
    velocity = StaggeredGrid(0, extrapolation.BOUNDARY, bounds=bounds, resolution=spatial(x=resolution[0], y=resolution[1], z=resolution[2]))
    
    # Pressure
    pressure = CenteredGrid(0, extrapolation.BOUNDARY, bounds=bounds, resolution=spatial(x=resolution[0], y=resolution[1], z=resolution[2]))
    
    # Smoke (Scalar)
    smoke = CenteredGrid(0, extrapolation.BOUNDARY, bounds=bounds, resolution=spatial(x=resolution[0], y=resolution[1], z=resolution[2]))
    
    return velocity, pressure, smoke, bounds

def step_simulation(velocity, pressure, smoke, obstacle_mask, dt=1.0, diffusion_rate=0.01):
    # Advect velocity and smoke
    velocity = advect.semi_lagrangian(velocity, velocity, dt)
    smoke = advect.semi_lagrangian(smoke, velocity, dt)
    
    # Add inflow velocity (West)
    inflow = StaggeredGrid(1.0, extrapolation.BOUNDARY, bounds=velocity.bounds, resolution=velocity.resolution)
    velocity += inflow * 0.1
    
    # Add smoke source along West boundary
    source_box = Box(x=(0, 5), y=(0, 200), z=(0, 2))
    smoke_source = CenteredGrid(source_box, extrapolation.BOUNDARY, bounds=smoke.bounds, resolution=smoke.resolution)
    smoke += smoke_source * 0.1

    
    # Diffusion
    smoke = diffuse.explicit(smoke, diffusion_rate, dt)
    
    # Pressure projection
    velocity, pressure = fluid.make_incompressible(velocity, ())
    
    # Apply obstacle no-slip condition
    staggered_mask = phi.field.StaggeredGrid(obstacle_mask, extrapolation.BOUNDARY, bounds=velocity.bounds, resolution=velocity.resolution)
    velocity = velocity * (1 - staggered_mask)
    
    # Vorticity
    # curl() might need to be imported or called differently
    # Let's try calling it as a function if imported, or field.curl
    vorticity = curl(velocity)
    
    return velocity, pressure, smoke, vorticity

