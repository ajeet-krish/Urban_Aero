from phi.torch.flow import *
from src.simulation import setup_simulation, step_simulation
from src.geometry import create_mask
from src.visualization import create_animation
from phi import math

def run_layout(layout_type):
    print(f"Running simulation for layout: {layout_type}")
    res = (64, 64, 16)
    domain_size = (200, 200, 50)
    
    velocity, pressure, smoke, bounds = setup_simulation(resolution=res, domain_size=domain_size)
    obstacle_mask = create_mask(domain_size, res, layout_type=layout_type)
    
    frames = []
    for i in range(50):
        velocity, pressure, smoke, vorticity = step_simulation(velocity, pressure, smoke, obstacle_mask)
        if i % 5 == 0:
            # Capture magnitude of velocity at z=2
            vel_centered = velocity.at_centers()
            
            # Vorticity needs to be a scalar field for visualization
            # Resample to a centered grid of the target resolution explicitly
            target_grid = CenteredGrid(0, extrapolation.BOUNDARY, bounds=bounds, resolution=spatial(x=res[0], y=res[1], z=res[2]))
            vort_centered = vorticity.at(target_grid)
            vort_scalar = math.vec_length(vort_centered.values)
            vort_grid = CenteredGrid(vort_scalar, extrapolation.BOUNDARY, bounds=bounds, resolution=spatial(x=res[0], y=res[1], z=res[2]))
            
            frames.append((vel_centered, vort_grid, smoke))
            print(f"Step {i} captured.")



        
    create_animation(frames, obstacle_mask, filename=f"assets/{layout_type}_sim.gif")
    print(f"Finished {layout_type}")

if __name__ == "__main__":
    for layout in ["grid", "staggered", "canyon", "courtyard"]:
        run_layout(layout)
