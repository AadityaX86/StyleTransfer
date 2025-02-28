import os
import shutil

# Define the source and destination directories
source_dir = "Source Dir"
destination_dir = "Destination Dir"

# Ensure destination folder exists
os.makedirs(destination_dir, exist_ok=True)

for i in range(0,0):
    # Create the filename for each image
    filename = f"{i}._"
    
    # Define full paths for source and destination files
    source_file = os.path.join(source_dir, filename)
    destination_file = os.path.join(destination_dir, filename)
    
    # Check if the file exists in the source directory before copying
    if os.path.exists(source_file):
        shutil.copy(source_file, destination_file)
        print(f"Copied {filename}")
    else:
        print(f"{filename} does not exist in the source directory.")

print("Copying complete.")
