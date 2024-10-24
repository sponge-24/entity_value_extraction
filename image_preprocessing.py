from PIL import Image, ImageEnhance
import cv2
import os
import numpy as np
from multiprocessing import Pool, cpu_count

def enhance_contrast(image, factor=2):
    """Enhance contrast of a PIL image."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def noise_removal(image):
    """Remove noise from an image using dilation and erosion."""
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    return image

def process_single_image(args):
    """Process a single image, enhance contrast and remove noise."""
    input_path, output_path, factor = args
    
    try:
        # Open the image using PIL and enhance contrast
        image = Image.open(input_path)
        enhanced_image = enhance_contrast(image, factor)

        # Convert to OpenCV format (numpy array)
        cv2_image = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)

        # Remove noise
        final_image = noise_removal(cv2_image)

        # Save the final processed image
        cv2.imwrite(output_path, final_image)
        print(f"Processed {os.path.basename(input_path)}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def process_images_in_directory(input_dir, output_dir, factor=2):
    """Process all images in the input directory using multiprocessing."""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare the list of image paths to process
    image_paths = [
        (os.path.join(input_dir, filename), os.path.join(output_dir, filename), factor)
        for filename in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, filename))
    ]

    # Determine the number of processes to use (equal to the number of CPU cores)
    num_workers = cpu_count()
    print(f"Using {num_workers} parallel workers.")

    # Use a Pool to process images in parallel
    with Pool(processes=num_workers) as pool:
        pool.map(process_single_image, image_paths)

# Example usage
if __name__ == "__main__":
    input_directory = "./test"  # Directory containing the input images
    output_directory = "./test_preprocessed"  # Directory where processed images will be saved

    # Process all images in the input directory and save them in the output directory
    process_images_in_directory(input_directory, output_directory)
