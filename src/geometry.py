from phi.torch.flow import *

def create_mask(domain_size, resolution, layout_type="grid"):
    bounds = Box(x=domain_size[0], y=domain_size[1], z=domain_size[2])
    mask = CenteredGrid(0, extrapolation.BOUNDARY, bounds=bounds, resolution=spatial(x=resolution[0], y=resolution[1], z=resolution[2]))
    
    building_w = 40
    
    if layout_type == "grid":
        spacing = 20
        for i in range(3):
            for j in range(3):
                x_pos = 20 + i * (building_w + spacing)
                y_pos = 20 + j * (building_w + spacing)
                box = Box(x=(x_pos, x_pos+building_w), y=(y_pos, y_pos+building_w), z=(0, 20))
                mask = mask + CenteredGrid(box, extrapolation.BOUNDARY, bounds=bounds, resolution=mask.resolution)

    elif layout_type == "staggered":
        spacing = 20
        for i in range(3):
            for j in range(3):
                x_pos = 20 + i * (building_w + spacing)
                # Offset middle row
                y_shift = 15 if i == 1 else 0
                y_pos = 20 + j * (building_w + spacing) + y_shift
                box = Box(x=(x_pos, x_pos+building_w), y=(y_pos, y_pos+building_w), z=(0, 20))
                mask = mask + CenteredGrid(box, extrapolation.BOUNDARY, bounds=bounds, resolution=mask.resolution)

    elif layout_type == "canyon":
        # Two long blocks
        block1 = Box(x=(40, 80), y=(20, 180), z=(0, 20))
        block2 = Box(x=(120, 160), y=(20, 180), z=(0, 20))
        mask = mask + CenteredGrid(block1, extrapolation.BOUNDARY, bounds=bounds, resolution=mask.resolution)
        mask = mask + CenteredGrid(block2, extrapolation.BOUNDARY, bounds=bounds, resolution=mask.resolution)

    elif layout_type == "courtyard":
        # Perimeter block
        perimeter = Box(x=(40, 160), y=(40, 160), z=(0, 20))
        void = Box(x=(80, 120), y=(80, 120), z=(0, 20))
        mask = mask + CenteredGrid(perimeter, extrapolation.BOUNDARY, bounds=bounds, resolution=mask.resolution)
        mask = mask - CenteredGrid(void, extrapolation.BOUNDARY, bounds=bounds, resolution=mask.resolution)

    return mask
