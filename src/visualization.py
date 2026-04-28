import os
from pathlib import Path
from phi import vis as phi_vis
import subprocess

def create_animated_quiver(
    velocity_trajectory,
    smoke_trajectory,
    obstacle_mask,
    output_dir="assets",
    filename_prefix="simulation",
    fps=10,
    z_slice=2,
    arrow_density="mixed"
):
    Path(output_dir).mkdir(exist_ok=True)
    
    vel_2d = velocity_trajectory.z[z_slice]
    
    mp4_path = os.path.join(output_dir, f"{filename_prefix}.mp4")
    gif_path = os.path.join(output_dir, f"{filename_prefix}.gif")
    
    try:
        ani = phi_vis.plot(vel_2d, animate='time')
        ani.save(mp4_path, writer='ffmpeg')
        
        subprocess.run(["ffmpeg", "-y", "-i", mp4_path, "-vf", "fps=10,scale=320:-1:flags=lanczos", "-c:v", "gif", gif_path], check=True)
        print(f"✓ MP4: {mp4_path}")
        print(f"✓ GIF: {gif_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
