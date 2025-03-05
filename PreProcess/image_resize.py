import os
from PIL import Image

def resize_images(input_folder, output_folder, target_size=(224, 224)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img.save(os.path.join(output_folder, filename))

if __name__ == "__main__":
    input_folder = r'Source Directory'# Path to the folder containing images to resize
    output_folder = r'Destination Directory'# Path to the folder to save resized images
    resize_images(input_folder, output_folder)