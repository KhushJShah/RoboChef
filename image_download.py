
import os

from google_images_download import google_images_download


def download_images(keyword, limit=10, output_directory='downloads'):
    # Initialize the response object
    response = google_images_download.googleimagesdownload()
    
    # Create the image directory for the keyword
    image_directory = keyword.replace(" ", "_")  # Replace spaces with underscores for folder names
    
    # Define the arguments for the search
    arguments = {
        "keywords": keyword,
        "limit": limit,  # Number of images to download
        "print_urls": True,
        "output_directory": output_directory,
        "image_directory": image_directory
    }
    
    # Download the images
    paths = response.download(arguments)
    
    # Print the paths of the downloaded images
    print(f"Downloaded {keyword} images to: {os.path.join(output_directory, image_directory)}")
    print(paths)

# Example usage
download_images(keyword="cinnamon sticks", limit=10, output_directory='dataset')
download_images(keyword="cloves", limit=10, output_directory='dataset')

