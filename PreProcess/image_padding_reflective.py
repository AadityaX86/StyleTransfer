from PIL import Image, ImageOps
import numpy as np
import os

def preprocess_image_with_reflective_padding(image_path, target_size=256, output_dir=None):
    try:
        # Open the image and ensure it's not corrupted
        img = Image.open(image_path).convert("RGB")  # Ensure 3 channels (RGB)
        
        # Get original dimensions
        original_width, original_height = img.size
        
        # Calculate aspect ratio
        aspect_ratio = original_width / original_height
        
        if aspect_ratio > 1:  # Width > Height
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:  # Height >= Width
            new_height = target_size
            new_width = int(target_size * aspect_ratio)
        
        # Resize image while maintaining aspect ratio
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Calculate padding to make the image square
        pad_left = (target_size - new_width) // 2
        pad_top = (target_size - new_height) // 2
        pad_right = target_size - new_width - pad_left
        pad_bottom = target_size - new_height - pad_top

        # Convert image to NumPy array
        img_array = np.array(img)

        # Apply reflective padding using NumPy
        img_padded = np.pad(
            img_array,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),  # Padding for (height, width, channels)
            mode='reflect'
        )

        # Convert back to PIL Image
        img = Image.fromarray(img_padded)

        # Save the preprocessed image if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
            filename = os.path.basename(image_path)
            img.save(os.path.join(output_dir, filename))

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

image_dir = r"Source Dir"  # Directory containing original images
output_dir = r"Destination Dir"  # Directory to save processed images

processed_images = []

for filename in os.listdir(image_dir):
    image_path = os.path.join(image_dir, filename)
    processed_img = preprocess_image_with_reflective_padding(image_path, output_dir=output_dir)
    if processed_img is not None:
        processed_images.append(processed_img)

print(f"Processed and saved {len(processed_images)} images with reflective padding to '{output_dir}'.")
