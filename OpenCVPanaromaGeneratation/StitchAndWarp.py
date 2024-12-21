import cv2
import numpy as np
import os

def normalize_16bit_to_uint8(image):
    """Normalize a 16-bit image to uint8."""
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized_image.astype(np.uint8)

def compute_projected_extent(image_shape, homography):
    """
    Compute the projected extent of the current image using its right two corners.

    Args:
        image_shape (tuple): The (height, width) of the image.
        homography (np.ndarray): The homography matrix.

    Returns:
        int: The maximum x-coordinate of the projected corners.
    """
    height, width = image_shape[:2]
    
    # Define the right two corners of the image
    corners = np.array([[width - 1, 0], [width - 1, height - 1]], dtype=np.float32)
    corners = corners.reshape(-1, 1, 2)
    
    # Transform the corners using the homography
    projected_corners = cv2.perspectiveTransform(corners, homography)
    
    # Extract x-coordinates of the transformed corners
    x_coords = projected_corners[:, 0, 0]
    
    # Return the maximum x-coordinate
    return int(np.ceil(x_coords.min()))

def stitch_images(input_folder, output_folder, output_file):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Read all image files from the input folder
    image_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.tif'))])
    images = [cv2.imread(img, cv2.IMREAD_UNCHANGED) for img in image_files]
    
    if not images:
        raise ValueError("No images found in the input folder!")
    
    # Normalize images to uint8
    images = [normalize_16bit_to_uint8(img) for img in images]
    
    # Initialize panorama with the first image
    panorama = images[0]
    cv2.imwrite(os.path.join(output_folder, "intermediate_0.tif"), panorama)
    
    for i in range(1, len(images)):
        # Get current image
        next_image = images[i]
        
        # Extract overlapping regions
        overlap1 = panorama[:, -512:]  # Last 512 columns of the current panorama
        overlap2 = next_image[:, :512]  # First 512 columns of the next image
        
        # Detect features using SIFT
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(overlap1, None)
        kp2, des2 = sift.detectAndCompute(overlap2, None)
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimate homography
        H, mask = cv2.findHomography(dst_pts, src_pts + np.array([panorama.shape[1] - 512, 0]), cv2.RANSAC, 5.0)
        print(H)
        # Compute the projected extent for the right corners
        new_width = compute_projected_extent(next_image.shape, H)
        print(f"New width after projecting corners for image {i}: {new_width}")
        
        # Warp next image to the panorama
        h, w = panorama.shape[:2]
        panorama_warp = cv2.warpPerspective(next_image, H, (max(w, new_width), max(h, next_image.shape[0])))
        
        # Combine with the current panorama
        panorama_warp[0:h, 0:w] = panorama
        panorama = panorama_warp
        
        # Save intermediate panorama
        intermediate_path = os.path.join(output_folder, f"intermediate_{i}.tif")
        cv2.imwrite(intermediate_path, panorama)
        print(f"Saved intermediate panorama: {intermediate_path}")
    
    # Save final panorama
    cv2.imwrite(output_file, panorama)
    print(f"Final panorama saved at: {output_file}")

# Example usage
if __name__ == "__main__":
    input_folder = r"D:/Data/AyPers/214"
    output_folder = r"D:/Data/AyPers/214/panoramas"
    output_file = r"D:/Data/AyPers/214/panoramas/output_panorama.tif"
    
    stitch_images(input_folder, output_folder, output_file)
