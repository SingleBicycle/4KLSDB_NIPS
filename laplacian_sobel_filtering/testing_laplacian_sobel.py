import random
import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image

# Your existing functions
patch_size = 240
flat_threshold = 100  # You can adjust as needed

def sobel_variance(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    height, width = magnitude.shape

    total_processed_patch = 0
    flat_region_count = 0

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = magnitude[i:min(i + patch_size, height), 
                              j:min(j + patch_size, width)]
            total_processed_patch += 1
            var_sobel = np.var(patch)
            if var_sobel < flat_threshold:
                flat_region_count += 1

    flat_ratio = flat_region_count / total_processed_patch if total_processed_patch > 0 else 0
    # Return the ratio of patches that are below 'flat_threshold'
    return flat_ratio

def laplacian_variance(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# Main Script
if __name__ == "__main__":
    file_path = "/data/tzz/4k_data_curation/milestone_1_summary/filtered_images.csv"  # Path to your CSV with image URLs
    df = pd.read_csv(file_path, engine="python", dtype=str)
    
    # 1. Take a random subsample of 300 rows (adjust n as needed)
    sample_size = 300
    sample_df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    results = []

    for idx, row in sample_df.iterrows():
        url_link = row['url']
        try:
            response = requests.get(url_link, stream=True, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"Error downloading {url_link}: {e}")
            continue

        # Convert to grayscale
        im = Image.open(response.raw).convert("RGB")
        img_np = np.array(im)
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Compute sharpness/flatness metrics
        lap_var = laplacian_variance(gray_img)
        flat_ratio = sobel_variance(gray_img)  # Ratio of flat patches in the image

        # Store results (including the original row if you like)
        results.append({
            'index': idx,
            'url': url_link,
            'laplacian_var': lap_var,
            'flat_ratio': flat_ratio
        })

    # 2. Convert to a DataFrame
    analysis_df = pd.DataFrame(results)

    # 3. Sort by Laplacian variance (ascending) so you can inspect “lowest to highest”
    analysis_df.sort_values('laplacian_var', inplace=True)
    analysis_df.reset_index(drop=True, inplace=True)

    # 4. Save to a CSV for convenience and manual review
    analysis_df.to_csv("analysis_sample_4.csv", index=False)

    print("Analysis complete. Check 'analysis_sample_4.csv' to see results.")
