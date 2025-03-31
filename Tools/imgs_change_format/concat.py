import os
from PIL import Image
import numpy as np

def concatenate_images(image_folder, output_filename="concatenated_image.jpg"):
    """
    Concatenates images in a folder horizontally, assuming they have the same height.

    Args:
        image_folder: The path to the folder containing the images.
        output_filename: The name of the output file (e.g., "combined.jpg").
    """

    try:
        images = []
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # Add other extensions if needed.
                filepath = os.path.join(image_folder, filename)
                try:
                    img = Image.open(filepath)
                    images.append(np.array(img))  # Convert to NumPy array immediately
                except Exception as e:
                    print(f"Error opening or processing image {filename}: {e}")
                    return  # Or handle the error differently

        if not images:
            print("No valid images found in the folder.")
            return

        # Check if all images have the same height
        heights = [img.shape[0] for img in images]
        if len(set(heights)) != 1:  # Check if all heights are the same
            print("Images have different heights. Resizing to the height of the first image.")
            standard_height = heights[0]
            resized_images = []
            for img in images:
                original_image = Image.fromarray(img)  # Convert NumPy back to PIL for resizing
                resized_image = original_image.resize((int(original_image.width * (standard_height / original_image.height)), standard_height))
                resized_images.append(np.array(resized_image))
            images = resized_images # Use the resized images instead

        concatenated_image = np.concatenate(images, axis=1)

        # Convert back to PIL Image and save
        final_image = Image.fromarray(concatenated_image.astype(np.uint8)) # Important: Ensure correct data type
        final_image.save(output_filename)
        print(f"Concatenated image saved as {output_filename}")


    except FileNotFoundError:
        print(f"Error: Folder '{image_folder}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage:
folder_path = "D:/TUW/Tools/imgs_change_format/"  # Replace with the actual path
concatenate_images(folder_path, "combined_images.jpg")  # You can change the output filename