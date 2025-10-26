#!/bin/bash

# Download Sample Images Script
# This script downloads free sample images from Unsplash for testing

echo "🖼️  Downloading sample images for ViT Auditing Toolkit..."
echo ""

# Create directories if they don't exist
mkdir -p examples/{basic_explainability,counterfactual,calibration,bias_detection,general}

# Function to download image with progress
download_image() {
    local url=$1
    local output=$2
    local description=$3
    
    echo "📥 Downloading: $description"
    curl -L "$url" -o "$output" --progress-bar
    
    if [ $? -eq 0 ]; then
        echo "✅ Saved to: $output"
    else
        echo "❌ Failed to download: $description"
    fi
    echo ""
}

echo "=== Basic Explainability Images ==="
echo ""

# Cat portrait
download_image \
    "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=800&q=80" \
    "examples/basic_explainability/cat_portrait.jpg" \
    "Cat Portrait"

# Dog portrait
download_image \
    "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=800&q=80" \
    "examples/basic_explainability/dog_portrait.jpg" \
    "Dog Portrait"

# Bird in flight
download_image \
    "https://images.unsplash.com/photo-1444464666168-49d633b86797?w=800&q=80" \
    "examples/basic_explainability/bird_flying.jpg" \
    "Bird Flying"

# Sports car
download_image \
    "https://images.unsplash.com/photo-1583121274602-3e2820c69888?w=800&q=80" \
    "examples/basic_explainability/sports_car.jpg" \
    "Sports Car"

# Coffee cup
download_image \
    "https://images.unsplash.com/photo-1509042239860-f550ce710b93?w=800&q=80" \
    "examples/basic_explainability/coffee_cup.jpg" \
    "Coffee Cup"

echo "=== Counterfactual Analysis Images ==="
echo ""

# Face centered
download_image \
    "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=800&q=80" \
    "examples/counterfactual/face_portrait.jpg" \
    "Face Portrait (for patch analysis)"

# Car side view
download_image \
    "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=800&q=80" \
    "examples/counterfactual/car_side.jpg" \
    "Car Side View"

# Building architecture
download_image \
    "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=800&q=80" \
    "examples/counterfactual/building.jpg" \
    "Building Architecture"

# Simple object - flower
download_image \
    "https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=800&q=80" \
    "examples/counterfactual/flower.jpg" \
    "Flower (simple object)"

echo "=== Calibration Test Images ==="
echo ""

# High quality clear image
download_image \
    "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?w=800&q=80" \
    "examples/calibration/clear_panda.jpg" \
    "Clear High-Quality Image"

# Slightly challenging
download_image \
    "https://images.unsplash.com/photo-1425082661705-1834bfd09dca?w=800&q=80" \
    "examples/calibration/outdoor_scene.jpg" \
    "Outdoor Scene (medium difficulty)"

# Complex scene
download_image \
    "https://images.unsplash.com/photo-1519389950473-47ba0277781c?w=800&q=80" \
    "examples/calibration/workspace.jpg" \
    "Complex Workspace Scene"

echo "=== Bias Detection Images ==="
echo ""

# Day lighting
download_image \
    "https://images.unsplash.com/photo-1601758228041-f3b2795255f1?w=800&q=80" \
    "examples/bias_detection/dog_daylight.jpg" \
    "Dog in Daylight"

# Indoor lighting
download_image \
    "https://images.unsplash.com/photo-1596492784531-6e6eb5ea9993?w=800&q=80" \
    "examples/bias_detection/cat_indoor.jpg" \
    "Cat Indoors"

# Outdoor scene
download_image \
    "https://images.unsplash.com/photo-1530595467537-0b5996c41f2d?w=800&q=80" \
    "examples/bias_detection/bird_outdoor.jpg" \
    "Bird Outdoors"

# Urban environment
download_image \
    "https://images.unsplash.com/photo-1449844908441-8829872d2607?w=800&q=80" \
    "examples/bias_detection/urban_scene.jpg" \
    "Urban Environment"

echo "=== General Test Images ==="
echo ""

# Food
download_image \
    "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=800&q=80" \
    "examples/general/pizza.jpg" \
    "Pizza"

# Nature
download_image \
    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&q=80" \
    "examples/general/mountain.jpg" \
    "Mountain Landscape"

# Technology
download_image \
    "https://images.unsplash.com/photo-1593642632823-8f785ba67e45?w=800&q=80" \
    "examples/general/laptop.jpg" \
    "Laptop"

# Furniture
download_image \
    "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=800&q=80" \
    "examples/general/chair.jpg" \
    "Modern Chair"

echo ""
echo "======================================"
echo "✅ Download complete!"
echo "======================================"
echo ""
echo "📊 Summary:"
echo "  - Basic Explainability: $(ls examples/basic_explainability/*.jpg 2>/dev/null | wc -l) images"
echo "  - Counterfactual: $(ls examples/counterfactual/*.jpg 2>/dev/null | wc -l) images"
echo "  - Calibration: $(ls examples/calibration/*.jpg 2>/dev/null | wc -l) images"
echo "  - Bias Detection: $(ls examples/bias_detection/*.jpg 2>/dev/null | wc -l) images"
echo "  - General: $(ls examples/general/*.jpg 2>/dev/null | wc -l) images"
echo ""
echo "🚀 Ready to test! Run: python app.py"
echo ""
