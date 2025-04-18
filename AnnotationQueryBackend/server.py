# Import necessary libraries
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pathlib import Path
import faiss
from huggingface_hub import login

# Import DINO v2 model
from transformers import AutoImageProcessor, AutoModel

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load DINO v2 model - using ViT base with patch size 14
login(token = "")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Set to evaluation mode

# Setup image transformations
transform = T.Compose([
    T.Resize((224, 224)),  # Resize image to DINO v2 input size
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_db = {}

def normalize_region_for_feature_extraction(region_image):
    """
    Scale region to 224x224 while preserving aspect ratio with padding.
    This prevents distortion when resizing small regions.
    """
    width, height = region_image.size
    
    # Determine scaling factor
    scale = min(224/width, 224/height)
    new_width, new_height = int(width * scale), int(height * scale)
    
    # Resize while preserving aspect ratio
    region_resized = region_image.resize((new_width, new_height))
    
    # Create padded image (black background)
    padded_image = Image.new("RGB", (224, 224))
    
    # Center the region in the padded image
    paste_x = (224 - new_width) // 2
    paste_y = (224 - new_height) // 2
    padded_image.paste(region_resized, (paste_x, paste_y))
    
    return padded_image



def initialize_database(image_folder, scale_factors=[1.0, 0.66, 0.33]):
    """
    Initialize the image database with an image pyramid approach.
    Each image is stored at multiple scales to improve matching small regions.
    """
    print(f"Building image database from {image_folder}...")
    
    image_paths = list(Path(image_folder).glob("**/*.jpg")) + list(Path(image_folder).glob("**/*.png"))
    
    features = []
    paths = []
    metadata = []  # To store additional info like scale factor
    
    for img_idx, img_path in enumerate(image_paths):
        try:
            # Load image
            image = Image.open(img_path).convert("RGB")
            orig_width, orig_height = image.size
            
            # Process each scale factor
            for scale in scale_factors:
                if scale == 1.0:
                    # Original image
                    scaled_image = image
                else:
                    # Calculate new dimensions
                    new_width = int(orig_width * scale)
                    new_height = int(orig_height * scale)
                    
                    # Skip very small scales
                    if new_width < 64 or new_height < 64:
                        continue
                        
                    # Create scaled version
                    scaled_image = image.resize((new_width, new_height))
                
                # Extract features from this scaled version
                try:
                    feature = extract_features(scaled_image)
                    
                    # Add to our collections
                    features.append(feature)
                    paths.append(str(img_path))
                    metadata.append({
                        "original_path": str(img_path),
                        "scale": scale,
                        "width": scaled_image.width,
                        "height": scaled_image.height
                    })
                except Exception as e:
                    print(f"Error extracting features from scaled image {img_path} at scale {scale}: {e}")
            
            if (img_idx + 1) % 100 == 0:
                print(f"Processed {img_idx + 1}/{len(image_paths)} images ({len(features)} total entries)")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    if not features:
        print("No features extracted! Check your image folder and image formats.")
        return
        
    # Convert to numpy array
    features_array = np.array(features).astype('float32')
    
    # Create FAISS index for fast similarity search
    d = features_array.shape[1]  # Dimension of feature vectors
    index = faiss.IndexFlatIP(d)  # Inner product similarity (equivalent to cosine for normalized vectors)
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(features_array)
    
    # Add to index
    index.add(features_array)
    
    # Update the global image_db
    image_db = {
        "features": features_array,
        "paths": paths,
        "metadata": metadata,
        "index": index
    }
    
    print(f"Database initialized with {len(paths)} entries from {len(image_paths)} images")
    return image_db

def extract_features(image):
    """
    Extract features from an image using DINO v2.
    Now enhanced to better handle small regions.
    """
    # Check if this is a small region
    width, height = image.size
    is_small_region = width < 100 or height < 100
    
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # For small regions, consider enhancing contrast to make features stand out
    if is_small_region:
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)  # Slightly boost contrast
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Extract features using DINO v2
    with torch.no_grad():
        outputs = model(image_tensor)
        # Use CLS token features
        feature = outputs.last_hidden_state[:, 0].cpu().numpy().astype(np.float32)
    
    return feature.flatten()

def size_aware_matching(query_image, image_db, top_k=8):
    """
    Find similar images with awareness of region size and scale.
    """
    # Get query dimensions
    query_width, query_height = query_image.size
    query_area = query_width * query_height
    
    # Normalize region to preserve aspect ratio
    normalized_query = normalize_region_for_feature_extraction(query_image)
    
    # Extract features from normalized image
    query_feature = extract_features(normalized_query)
    
    # Normalize for cosine similarity
    query_feature = query_feature.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_feature)
    
    # Search for more candidates than needed
    expanded_k = min(top_k * 3, len(image_db["paths"]))
    D, I = image_db["index"].search(query_feature, expanded_k)
    
    # Process results with size adjustment
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(image_db["paths"]):
            path = image_db["paths"][idx]
            meta = image_db["metadata"][idx] if "metadata" in image_db else None
            
            # Get scale factor if available
            scale = meta["scale"] if meta and "scale" in meta else 1.0
            
            try:
                # Get original image dimensions
                with Image.open(path) as img:
                    img_width, img_height = img.size
                    
                    # For scaled entries, adjust the effective area
                    effective_width = img_width * scale
                    effective_height = img_height * scale
                    effective_area = effective_width * effective_height
                    
                    # Calculate size ratio
                    size_ratio = query_area / effective_area
                    
                    # Adjust score based on size ratio
                    if size_ratio < 0.05:
                        adjusted_score = score * (0.3 + 0.7 * size_ratio * 20)
                    elif size_ratio < 0.2:
                        adjusted_score = score * (0.7 + 0.3 * size_ratio * 5)
                    else:
                        adjusted_score = score
                    
                    # Boost scores for downscaled images when query is small
                    if query_area < 10000 and scale < 1.0:
                        # Boost more for smaller regions
                        scale_boost = 1.0 + (1.0 - scale) * 0.2
                        adjusted_score *= scale_boost
                    
                    results.append({
                        "path": path,
                        "raw_similarity": float(score),
                        "adjusted_similarity": float(adjusted_score),
                        "size_ratio": float(size_ratio),
                        "scale": float(scale)
                    })
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results.append({
                    "path": path,
                    "raw_similarity": float(score),
                    "adjusted_similarity": float(score),
                    "size_ratio": 1.0,
                    "scale": float(scale) if meta and "scale" in meta else 1.0
                })
    
    # Sort by adjusted similarity
    results.sort(key=lambda x: x["adjusted_similarity"], reverse=True)
    
    # Take top_k, removing duplicates (same image at different scales)
    unique_results = []
    seen_paths = set()
    
    for result in results:
        # Extract original path without scale information
        original_path = result["path"]
        
        if original_path not in seen_paths:
            seen_paths.add(original_path)
            unique_results.append(result)
            
            if len(unique_results) >= top_k:
                break
    
    return unique_results

@app.route('/api/process-region', methods=['POST'])
def process_region():
    """Process a region and find similar images with size-aware matching."""
    if 'region_image' not in request.files:
        return jsonify({"error": "No region_image provided"}), 400
    
    try:
        # Read the image
        region_file = request.files['region_image']
        region_image = Image.open(io.BytesIO(region_file.read()))
        
        # Find similar images with size-aware matching
        similar_images = size_aware_matching(region_image, image_db, top_k=8)
        
        # Prepare response
        response = {
            "success": True,
            "query_size": {
                "width": region_image.width,
                "height": region_image.height
            },
            "results": []
        }
        
        # Process results
        for result in similar_images:
            img_path = result["path"]
            try:
                with open(img_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    response["results"].append({
                        "image_data": f"data:image/jpeg;base64,{img_data}",
                        "similarity": result["adjusted_similarity"],
                        "raw_similarity": result["raw_similarity"],
                        "size_ratio": result["size_ratio"],
                        "path": os.path.basename(img_path)
                    })
            except Exception as e:
                print(f"Error processing result image {img_path}: {e}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in process_region: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Initialize the database at startup
if __name__ == "__main__":
    # image folder path
    IMAGE_FOLDER = "/home/parisol/Downloads/Img_annot/Img_annot/images/"
    image_db = initialize_database(IMAGE_FOLDER)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', use_reloader=False, port=5000)