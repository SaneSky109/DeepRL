import imageio
import os
import re

base_dir = "C:/Users/ericl/OneDrive/Documents/ReinforcementLearning"  # Change this to your specific path
folder_name = "Screenshots"
folder_path = os.path.join(base_dir, folder_name)

# Create the directory if it does not exist
if not os.path.isdir(folder_path):
    os.makedirs(folder_path, exist_ok=True)





def numerical_sort(value):
    """
    This function extracts numbers from the filename and uses them for sorting.
    """
    parts = re.compile(r'(\d+)').split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def create_gif_from_screenshots(screenshot_dir, output_gif_path, loop=0):
    """
    This function creates a GIF from screenshots stored in a directory,
    ensuring the files are sorted numerically based on the digits in their filenames.
    """
    screenshots = sorted((os.path.join(screenshot_dir, f) for f in os.listdir(screenshot_dir) if f.endswith('.png')), key=numerical_sort)
    # Create a GIF from the sorted list of screenshots.
    with imageio.get_writer(output_gif_path, mode='I', fps = 6, loop=loop) as writer:
        for filename in screenshots:
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f"GIF saved to {output_gif_path}")


output_gif_path = "C:/Users/ericl/OneDrive/Documents/ReinforcementLearning/GIFS/DQNAgent_gameplay.gif"
create_gif_from_screenshots(screenshot_dir = folder_path, output_gif_path = output_gif_path)