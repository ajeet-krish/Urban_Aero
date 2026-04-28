from src.simulation import setup_simulation, step_simulation
from src.geometry import create_urban_block_mask
from src.visualization import plot_slice
from src.export import export_fields

def run_sim():
    print("Initializing simulation...")
    res = (64, 64, 16)
    domain_size = (200, 200, 50)
    
    velocity, pressure, smoke, bounds = setup_simulation(resolution=res, domain_size=domain_size)
    obstacle_mask = create_urban_block_mask(domain_size, res)
    
    print("Running 50 steps...")
    for i in range(50):
        velocity, pressure, smoke = step_simulation(velocity, pressure, smoke, obstacle_mask)
        if i % 10 == 0:
            print(f"Step {i} complete.")
        
    print("Simulation complete. Exporting and plotting...")
    
    # Export final state
    export_fields(velocity, smoke, step=49)
    
    # Visualize pedestrian level
    velocity_centered = velocity.at_centers()
    plot_slice(velocity_centered.vector['x'], z_slice=2, title="Wind Velocity (X) at Pedestrian Level")
    plot_slice(smoke, z_slice=2, title="Smoke Concentration at Pedestrian Level")




if __name__ == "__main__":
    run_sim()
