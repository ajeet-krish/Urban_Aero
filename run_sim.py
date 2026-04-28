from src.simulation import setup_simulation, step_simulation, accumulate_trajectory
from src.geometry import create_mask
from src.visualization import create_animated_quiver
from phi.torch.flow import *
from phi import math

def run_sim():
    print("Initializing simulation...")
    res = (64, 64, 16)
    domain_size = (200, 200, 50)
    
    velocity, pressure, smoke, bounds = setup_simulation(resolution=res, domain_size=domain_size)
    obstacle_mask = create_mask(domain_size, res, layout_type="grid")
    
    centered_mask = CenteredGrid(obstacle_mask, extrapolation.BOUNDARY, bounds=bounds, resolution=spatial(x=res[0], y=res[1], z=res[2]))
    
    print("Running simulation and capturing frames...")
    velocity_frames = []
    smoke_frames = []
    
    for i in range(50):
        velocity, pressure, smoke, vorticity = step_simulation(velocity, pressure, smoke, obstacle_mask)
        if i % 5 == 0:
            velocity_frames.append(velocity)
            smoke_frames.append(smoke)
            print(f"Step {i} captured.")
        
    print("Exporting animation...")
    velocity_traj, smoke_traj = accumulate_trajectory(velocity_frames, smoke_frames)
    
    create_animated_quiver(
        velocity_traj, 
        smoke_traj, 
        obstacle_mask,
        output_dir="assets",
        filename_prefix="simulation",
        fps=10,
        z_slice=2
    )
    
    print("Simulation complete.")

if __name__ == "__main__":
    run_sim()






if __name__ == "__main__":
    run_sim()
