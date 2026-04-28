from phi.torch.flow import *
import torch
import phi.field
from phi.field import StaggeredGrid, CenteredGrid, curl

def setup_simulation(resolution=(64, 64, 16), domain_size=(200, 200, 50)):
    math.set_global_precision(32)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    bounds = Box(x=domain_size[0], y=domain_size[1], z=domain_size[2])
    
    velocity = StaggeredGrid(0, extrapolation.BOUNDARY, bounds=bounds, resolution=spatial(x=resolution[0], y=resolution[1], z=resolution[2]))
    pressure = CenteredGrid(0, extrapolation.BOUNDARY, bounds=bounds, resolution=spatial(x=resolution[0], y=resolution[1], z=resolution[2]))
    smoke = CenteredGrid(0, extrapolation.BOUNDARY, bounds=bounds, resolution=spatial(x=resolution[0], y=resolution[1], z=resolution[2]))
    
    return velocity, pressure, smoke, bounds

def accumulate_trajectory(velocity_frames, smoke_frames):
    from phi import math
    centered_velocity_frames = [v.at_centers() for v in velocity_frames]
    
    velocity_traj = math.stack(centered_velocity_frames, math.batch('time'))
    smoke_traj = math.stack(smoke_frames, math.batch('time'))
    
    return velocity_traj, smoke_traj

def step_simulation(velocity, pressure, smoke, obstacle_mask, dt=1.0, diffusion_rate=0.01):
    velocity = advect.semi_lagrangian(velocity, velocity, dt)
    smoke = advect.semi_lagrangian(smoke, velocity, dt)
    
    inflow = StaggeredGrid(1.0, extrapolation.BOUNDARY, bounds=velocity.bounds, resolution=velocity.resolution)
    velocity += inflow * 0.1
    
    source_box = Box(x=(0, 5), y=(0, 200), z=(0, 2))
    smoke_source = CenteredGrid(source_box, extrapolation.BOUNDARY, bounds=smoke.bounds, resolution=smoke.resolution)
    smoke += smoke_source * 0.1
    
    smoke = diffuse.explicit(smoke, diffusion_rate, dt)
    
    velocity, pressure = fluid.make_incompressible(velocity, ())
    
    staggered_mask = phi.field.StaggeredGrid(obstacle_mask, extrapolation.BOUNDARY, bounds=velocity.bounds, resolution=velocity.resolution)
    velocity = velocity * (1 - staggered_mask)
    
    vorticity = curl(velocity)
    
    return velocity, pressure, smoke, vorticity

