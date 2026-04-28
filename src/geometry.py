from phi.torch.flow import *

def create_urban_block_mask(domain_size, resolution):
    """
    Creates a 3x3 grid of buildings as obstacles within the domain.
    Returns a CenteredGrid mask (1 = building, 0 = air).
    """
    bounds = Box(x=domain_size[0], y=domain_size[1], z=domain_size[2])
    mask = CenteredGrid(0, extrapolation.BOUNDARY, bounds=bounds, resolution=spatial(x=resolution[0], y=resolution[1], z=resolution[2]))
    
    # Define building parameters
    building_w = 40
    spacing = 20
    
    # Create 3x3 grid
    for i in range(3):
        for j in range(3):
            # Calculate positions (simple grid)
            x_pos = 20 + i * (building_w + spacing)
            y_pos = 20 + j * (building_w + spacing)
            
            # Buildings of varying heights
            height = 10 + (i + j) * 5
            
            building_box = Box(x=(x_pos, x_pos+building_w), 
                               y=(y_pos, y_pos+building_w), 
                               z=(0, height))
            
            # CenteredGrid can take a Geometry as 'values' to rasterize it
            building_field = CenteredGrid(building_box, extrapolation.BOUNDARY, bounds=bounds, resolution=spatial(x=resolution[0], y=resolution[1], z=resolution[2]))
            mask = mask + building_field
            
    # Convert Centered mask to Staggered for velocity masking
    return StaggeredGrid(mask, extrapolation.BOUNDARY, bounds=bounds, resolution=spatial(x=resolution[0], y=resolution[1], z=resolution[2]))

