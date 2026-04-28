from src.simulation import setup_simulation, step_simulation
from src.geometry import create_urban_block_mask
from src.visualization import create_animation
from src.export import export_fields
from phi.torch.flow import *
from src.simulation import setup_simulation, step_simulation
from src.geometry import create_urban_block_mask
from src.visualization import create_animation
from src.export import export_fields
from phi import math

def run_sim():

    print("Initializing simulation...")
    res = (64, 64, 16)
    domain_size = (200, 200, 50)
    
    velocity, pressure, smoke, bounds = setup_simulation(resolution=res, domain_size=domain_size)
    obstacle_mask = create_urban_block_mask(domain_size, res)
    
    # Pre-calculate centered mask for visualization (z-slice 2)
    # The obstacle_mask is staggered, let's just make a centered one for viz
    centered_mask = CenteredGrid(obstacle_mask, extrapolation.BOUNDARY, bounds=bounds, resolution=spatial(x=res[0], y=res[1], z=res[2]))
    mask_slice = centered_mask.values.numpy('x,y,z')[:, :, 2]
    
    print("Running simulation and capturing frames...")
    frames = []
    for i in range(50):
        velocity, pressure, smoke = step_simulation(velocity, pressure, smoke, obstacle_mask)
        if i % 5 == 0:
            # Capture magnitude of velocity at z=2
            vel_centered = velocity.at_centers()
            # Use math.vec_length(field.values) which is the correct way
            # based on previous error
            vel_mag = math.vec_length(vel_centered.values)
            
            # Create a new CenteredGrid for viz from the magnitude
            vel_mag_grid = CenteredGrid(vel_mag, extrapolation.BOUNDARY, bounds=bounds, resolution=spatial(x=res[0], y=res[1], z=res[2]))
            
            frames.append((vel_mag_grid, smoke))
            print(f"Step {i} captured.")


        
    print("Exporting animation...")
    create_animation(frames, centered_mask, filename="assets/simulation.gif")
    
    print("Simulation complete.")

if __name__ == "__main__":
    run_sim()




if __name__ == "__main__":
    run_sim()
