from phi.torch.flow import *
from src.simulation import setup_simulation, step_simulation, accumulate_trajectory
from src.geometry import create_mask
from src.visualization import create_animated_quiver
from phi import math

def run_layout(layout_type):
    print(f"Running simulation for layout: {layout_type}")
    res = (64, 64, 16)
    domain_size = (200, 200, 50)
    
    velocity, pressure, smoke, bounds = setup_simulation(resolution=res, domain_size=domain_size)
    obstacle_mask = create_mask(domain_size, res, layout_type=layout_type)
    
    velocity_frames = []
    smoke_frames = []
    for i in range(50):
        velocity, pressure, smoke, vorticity = step_simulation(velocity, pressure, smoke, obstacle_mask)
        if i % 5 == 0:
            velocity_frames.append(velocity)
            smoke_frames.append(smoke)
            print(f"Step {i} captured.")
            
    velocity_traj, smoke_traj = accumulate_trajectory(velocity_frames, smoke_frames)
    
    create_animated_quiver(
        velocity_traj, 
        smoke_traj, 
        obstacle_mask,
        output_dir="assets",
        filename_prefix=f"{layout_type}_sim",
        fps=10,
        z_slice=2
    )
    print(f"Finished {layout_type}")

if __name__ == "__main__":
    for layout in ["grid", "staggered", "canyon", "courtyard"]:
        run_layout(layout)

