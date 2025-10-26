"""
Download Sample Images for ViT Auditing Toolkit
This Python script downloads free sample images from Unsplash for testing.
"""

import os
import urllib.request
from pathlib import Path

# Color codes for terminal output
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"


def download_image(url, filepath, description):
    """Download an image from URL to filepath."""
    try:
        print(f"{BLUE}📥 Downloading:{RESET} {description}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Download the image
        urllib.request.urlretrieve(url, filepath)

        # Check if file was created
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / 1024  # KB
            print(f"{GREEN}✅ Saved:{RESET} {filepath} ({file_size:.1f} KB)\n")
            return True
        else:
            print(f"{RED}❌ Failed to save:{RESET} {filepath}\n")
            return False

    except Exception as e:
        print(f"{RED}❌ Error:{RESET} {str(e)}\n")
        return False


def main():
    """Main function to download all sample images."""
    print("🖼️  Downloading sample images for ViT Auditing Toolkit...\n")

    # Base directory
    base_dir = "examples"

    # Create directories
    directories = [
        "basic_explainability",
        "counterfactual",
        "calibration",
        "bias_detection",
        "general",
    ]

    for directory in directories:
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)

    # Image download list: (url, filepath, description)
    images = [
        # Basic Explainability
        (
            "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=800&q=80",
            f"{base_dir}/basic_explainability/cat_portrait.jpg",
            "Cat Portrait",
        ),
        (
            "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=800&q=80",
            f"{base_dir}/basic_explainability/dog_portrait.jpg",
            "Dog Portrait",
        ),
        (
            "https://images.unsplash.com/photo-1444464666168-49d633b86797?w=800&q=80",
            f"{base_dir}/basic_explainability/bird_flying.jpg",
            "Bird Flying",
        ),
        (
            "https://images.unsplash.com/photo-1583121274602-3e2820c69888?w=800&q=80",
            f"{base_dir}/basic_explainability/sports_car.jpg",
            "Sports Car",
        ),
        (
            "https://images.unsplash.com/photo-1509042239860-f550ce710b93?w=800&q=80",
            f"{base_dir}/basic_explainability/coffee_cup.jpg",
            "Coffee Cup",
        ),
        # Counterfactual Analysis
        (
            "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=800&q=80",
            f"{base_dir}/counterfactual/face_portrait.jpg",
            "Face Portrait",
        ),
        (
            "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=800&q=80",
            f"{base_dir}/counterfactual/car_side.jpg",
            "Car Side View",
        ),
        (
            "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=800&q=80",
            f"{base_dir}/counterfactual/building.jpg",
            "Building Architecture",
        ),
        (
            "https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=800&q=80",
            f"{base_dir}/counterfactual/flower.jpg",
            "Flower",
        ),
        # Calibration
        (
            "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?w=800&q=80",
            f"{base_dir}/calibration/clear_panda.jpg",
            "Clear Panda Image",
        ),
        (
            "https://images.unsplash.com/photo-1425082661705-1834bfd09dca?w=800&q=80",
            f"{base_dir}/calibration/outdoor_scene.jpg",
            "Outdoor Scene",
        ),
        (
            "https://images.unsplash.com/photo-1519389950473-47ba0277781c?w=800&q=80",
            f"{base_dir}/calibration/workspace.jpg",
            "Workspace Scene",
        ),
        # Bias Detection
        (
            "https://images.unsplash.com/photo-1601758228041-f3b2795255f1?w=800&q=80",
            f"{base_dir}/bias_detection/dog_daylight.jpg",
            "Dog in Daylight",
        ),
        (
            "https://images.unsplash.com/photo-1596492784531-6e6eb5ea9993?w=800&q=80",
            f"{base_dir}/bias_detection/cat_indoor.jpg",
            "Cat Indoors",
        ),
        (
            "https://images.unsplash.com/photo-1530595467537-0b5996c41f2d?w=800&q=80",
            f"{base_dir}/bias_detection/bird_outdoor.jpg",
            "Bird Outdoors",
        ),
        (
            "https://images.unsplash.com/photo-1449844908441-8829872d2607?w=800&q=80",
            f"{base_dir}/bias_detection/urban_scene.jpg",
            "Urban Environment",
        ),
        # General
        (
            "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=800&q=80",
            f"{base_dir}/general/pizza.jpg",
            "Pizza",
        ),
        (
            "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&q=80",
            f"{base_dir}/general/mountain.jpg",
            "Mountain Landscape",
        ),
        (
            "https://images.unsplash.com/photo-1593642632823-8f785ba67e45?w=800&q=80",
            f"{base_dir}/general/laptop.jpg",
            "Laptop",
        ),
        (
            "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=800&q=80",
            f"{base_dir}/general/chair.jpg",
            "Modern Chair",
        ),
    ]

    # Download all images
    successful = 0
    failed = 0

    print("=" * 50)
    print("Starting downloads...\n")

    for url, filepath, description in images:
        if download_image(url, filepath, description):
            successful += 1
        else:
            failed += 1

    # Summary
    print("=" * 50)
    print(f"{GREEN}✅ Download complete!{RESET}")
    print("=" * 50)
    print(f"\n📊 Summary:")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"\n📁 Image count by category:")

    for directory in directories:
        path = Path(base_dir) / directory
        image_count = len(list(path.glob("*.jpg")))
        print(f"  - {directory}: {image_count} images")

    print(f"\n🚀 Ready to test! Run: python app.py\n")


if __name__ == "__main__":
    main()
