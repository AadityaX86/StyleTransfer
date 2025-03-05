from PIL import Image, ImageOps
import numpy as np
import os

def preprocess_image(image_path, target_size=224, output_dir=None):
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
        
        # Add padding to make the image square
        img = ImageOps.pad(img, (target_size, target_size), color=(0, 0, 0))  # Black padding
        
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
    processed_img = preprocess_image(image_path, output_dir=output_dir)
    if processed_img is not None:
        processed_images.append(processed_img)

print(f"Processed and saved {len(processed_images)} images to '{output_dir}'.")
