from PIL import Image
import numpy as np
import os

def preprocess_image_with_resize_then_crop(image_path, target_size=224, output_dir=None):
    try:
        # Open the image and ensure it's not corrupted
        img = Image.open(image_path).convert("RGB")  # Ensure 3 channels (RGB)

        # Get original dimensions
        original_width, original_height = img.size

        # Calculate aspect ratio
        aspect_ratio = original_width / original_height

        # Resize the image while maintaining aspect ratio
        if aspect_ratio > 1:  # Width > Height
            new_width = int(target_size * aspect_ratio)
            new_height = target_size
        else:  # Height >= Width
            new_width = target_size
            new_height = int(target_size / aspect_ratio)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center crop to the desired resolution
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        img = img.crop((left, top, right, bottom))

        # Save the preprocessed image if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
            filename = os.path.basename(image_path)
            img.save(os.path.join(output_dir, filename))

        # Normalize pixel values to [0, 1] for in-memory processing (optional)
        img_array = np.array(img) / 255.0

        return img_array

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

image_dir = r"Source Dir"  # Directory containing original images
output_dir = r"Destination Dir"  # Directory to save processed images

processed_images = []

for filename in os.listdir(image_dir):
    image_path = os.path.join(image_dir, filename)
    processed_img = preprocess_image_with_resize_then_crop(image_path, output_dir=output_dir)
    if processed_img is not None:
        processed_images.append(processed_img)

print(f"Processed and saved {len(processed_images)} images with resize then crop to '{output_dir}'.")
