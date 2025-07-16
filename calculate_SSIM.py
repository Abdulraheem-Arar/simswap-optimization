import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(image_path1, image_path2):
    """
    Calculate SSIM between two saved images
    Args:
        image_path1: Path to first image
        image_path2: Path to second image
    Returns:
        ssim_score: Float between 0-1 (1 = identical)
    """
    # Read images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    # Convert to float32 and normalize to 0-1
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    
    # Calculate SSIM for each channel and average
    ssim_channels = []
    for channel in range(3):  # For BGR channels
        ssim_val = ssim(
            img1[:, :, channel],
            img2[:, :, channel],
            data_range=1.0,
            win_size=11,  # Optimal for standard image sizes
            gaussian_weights=True
        )
        ssim_channels.append(ssim_val)
    
    return np.mean(ssim_channels)

# Example usage:
if __name__ == "__main__":
    image1 = "/scratch/aa10947/SimSwap/output/result.jpg"
    image2 = "/scratch/aa10947/SimSwap/amp_protect.jpg"
    
    ssim_score = calculate_ssim(image1, image2)
    print(f"SSIM Score: {ssim_score:.4f}")
    print(f"First image path: {image1}")
    print(f"second image path: {image2}")
    # Quality interpretation
    if ssim_score >= 0.95:
        print("Quality: Excellent (visually indistinguishable)")
    elif ssim_score >= 0.90:
        print("Quality: Good (minor differences)")
    elif ssim_score >= 0.80:
        print("Quality: Fair (noticeable but acceptable)")
    else:
        print("Quality: Poor (significant degradation)")