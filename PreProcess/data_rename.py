import os
import shutil

# Define source and destination directories
source_dir = 'Source Dir'
destination_dir = 'Destination Dir'

# Ensure destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# List all files in source directory
files = [f for f in os.listdir(source_dir) if f.lower().endswith('._')]

start_number = 1

# Rename and move files
for idx, file in enumerate(files, start=start_number):
    # Define new file name
    new_file_name = f"{idx}._"
    
    # Get full path of the current and new files
    source_file = os.path.join(source_dir, file)
    destination_file = os.path.join(destination_dir, new_file_name)
    
    # Move and rename the file
    shutil.copy(source_file, destination_file)

print(f"Renaming and copying complete.")
