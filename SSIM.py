# import os
# import cv2
# import json
# import numpy as np
# from skimage.metrics import structural_similarity as ssim

# def compute_ssim_batch(content_dir, stylized_dir, output_json):
#     content_images = sorted(os.listdir(content_dir))
#     stylized_images = sorted(os.listdir(stylized_dir))

#     ssim_scores = {}

#     for content_img, stylized_img in zip(content_images, stylized_images):
#         content_path = os.path.join(content_dir, content_img)
#         stylized_path = os.path.join(stylized_dir, stylized_img)

#         # Read images
#         content = cv2.imread(content_path, cv2.IMREAD_COLOR)
#         stylized = cv2.imread(stylized_path, cv2.IMREAD_COLOR)

#         # Resize stylized image to match content image (if needed)
#         stylized = cv2.resize(stylized, (content.shape[1], content.shape[0]))

#         # Convert to grayscale
#         content_gray = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)
#         stylized_gray = cv2.cvtColor(stylized, cv2.COLOR_BGR2GRAY)

#         # Compute SSIM
#         score, _ = ssim(content_gray, stylized_gray, full=True)
#         ssim_scores[content_img] = score  # Store SSIM score with image name

#     # Compute average SSIM
#     avg_ssim = np.mean(list(ssim_scores.values()))

#     # Save results to JSON
#     result = {
#         "average_ssim": avg_ssim,
#         "ssim_per_image": ssim_scores
#     }
#     with open(output_json, "w") as f:
#         json.dump(result, f, indent=4)

#     print(f"Results saved to {output_json}")
#     return avg_ssim

# # Example usage
# content_folder = r"D:\College\III_II\MinorProject\StyleTransfer\Contents"
# stylized_folder = r"D:\College\III_II\MinorProject\StyleTransfer\Outputs\StyTr2"
# output_json_file = "ssim_results.json"

# average_ssim = compute_ssim_batch(content_folder, stylized_folder, output_json_file)
# print(f"Average SSIM: {average_ssim}")


### Single Image SSIM

import os
import cv2
import json
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_ssim_single_content(content_image, stylized_dir, output_json):
    stylized_images = sorted(os.listdir(stylized_dir))
    ssim_scores = {}

    # Read content image
    content = cv2.imread(content_image, cv2.IMREAD_COLOR)
    if content is None:
        raise FileNotFoundError(f"Content image not found: {content_image}")

    # Convert content image to grayscale
    content_gray = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)

    for stylized_img in stylized_images:
        stylized_path = os.path.join(stylized_dir, stylized_img)

        # Read stylized image
        stylized = cv2.imread(stylized_path, cv2.IMREAD_COLOR)
        if stylized is None:
            print(f"Skipping {stylized_img}: Unable to read image.")
            continue

        # Resize stylized image to match content image (if needed)
        stylized = cv2.resize(stylized, (content.shape[1], content.shape[0]))

        # Convert to grayscale
        stylized_gray = cv2.cvtColor(stylized, cv2.COLOR_BGR2GRAY)

        # Compute SSIM
        score, _ = ssim(content_gray, stylized_gray, full=True)
        ssim_scores[stylized_img] = score  # Store SSIM score with image name

    # Compute average SSIM across all stylized images
    avg_ssim = np.mean(list(ssim_scores.values()))

    # Save results to JSON
    result = {
        "content_image": os.path.basename(content_image),
        "average_ssim": avg_ssim,
        "ssim_per_image": ssim_scores
    }
    with open(output_json, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Results saved to {output_json}")
    return avg_ssim

# Example usage
content_image_path = r"D:\College\III_II\MinorProject\Report\content_leak\image_content.jpg"
stylized_folder = r"D:\College\III_II\MinorProject\Report\content_leak"
output_json_file = "ssim_results.json"

average_ssim = compute_ssim_single_content(content_image_path, stylized_folder, output_json_file)
print(f"Average SSIM: {average_ssim}")
