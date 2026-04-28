import matplotlib.pyplot as plt
from phi.torch.flow import *

def plot_slice(grid, z_slice=2, title="Field Slice"):
    """
    Plots a 2D slice of a 3D field at a specific Z index.
    """
    # Extract data. Grid.values has spatial dimensions
    # We need to specify the order 'x,y,z'
    data = grid.values.numpy('x,y,z')
    # If the grid has a batch/channel dimension, we should handle it
    # CenteredGrid values usually have (batch, channel, x, y, z)
    # The 'values' property might already be just spatial if it's a grid?
    # Let's inspect the shape in the plot function or just slice it.
    
    # If data is (x, y, z), then data[:, :, z_slice] is correct.
    # If data is (b, c, x, y, z), we need to select indices.
    
    if data.ndim == 5:
        data = data[0, 0, :, :, z_slice]
    elif data.ndim == 3:
        data = data[:, :, z_slice]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(data.T, origin='lower', cmap='viridis')
    plt.colorbar(label="Value")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

