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

# Create a database of feature vectors from your images
# This would be initialized at startup
image_db = {
    "features": [],  # List to store feature vectors
    "paths": [],     # List to store corresponding image paths
    "index": None    # FAISS index for fast similarity search
}

def initialize_database(image_folder):
    """Initialize the image database by extracting features from all images."""
    print(f"Building image database from {image_folder}...")
    
    image_paths = list(Path(image_folder).glob("**/*.jpg")) + list(Path(image_folder).glob("**/*.png"))
    
    features = []
    paths = []
    
    for img_path in image_paths:
        try:
            # Load and process image
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Could not open image {img_path}: {e}")
                continue
                
            # Apply transformations
            try:
                image_tensor = transform(image).unsqueeze(0).to(device)
            except Exception as e:
                print(f"Error transforming image {img_path}: {e}")
                continue
            
            # Extract features
            try:
                with torch.no_grad():
                    outputs = model(image_tensor)
                    # Make sure the feature is a numpy array with correct type
                    feature = outputs.last_hidden_state[:, 0].cpu().numpy().astype(np.float32)
            except Exception as e:
                print(f"Error extracting features from {img_path}: {e}")
                continue
            
            # Ensure feature is the right shape and type
            if not isinstance(feature, np.ndarray):
                print(f"Feature from {img_path} is not a numpy array, skipping")
                continue
                
            features.append(feature.flatten())
            paths.append(str(img_path))
            
            if len(paths) % 100 == 0:
                print(f"Processed {len(paths)} images")
                
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
    image_db["features"] = features_array
    image_db["paths"] = paths
    image_db["index"] = index
    
    print(f"Database initialized with {len(paths)} images")

def extract_features(image):
    """Extract features from an image using DINO v2."""
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Extract features using DINO v2
    with torch.no_grad():
        outputs = model(image_tensor)
        # Explicitly convert to numpy array with float32 type
        feature = outputs.last_hidden_state[:, 0].cpu().numpy().astype(np.float32)
    
    return feature.flatten()

def find_similar_images(query_feature, top_k=8):
    """Find similar images based on feature similarity."""
    # Normalize query vector for cosine similarity
    query_feature = query_feature.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_feature)
    
    # Search for similar vectors
    D, I = image_db["index"].search(query_feature, top_k)
    
    # Get results
    results = []
    for i, (score, idx) in enumerate(zip(D[0], I[0])):
        if idx < len(image_db["paths"]):  # Ensure valid index
            results.append({
                "path": image_db["paths"][idx],
                "similarity": float(score),
                "rank": i + 1
            })
    
    return results

@app.route('/api/process-region', methods=['POST'])
def process_region():
    """Process a region and find similar images."""
    if 'region_image' not in request.files:
        return jsonify({"error": "No region_image provided"}), 400
    
    try:
        # Read the image
        region_file = request.files['region_image']
        region_image = Image.open(io.BytesIO(region_file.read()))
        
        # Extract features from the region
        query_feature = extract_features(region_image)
        
        # Find similar images
        similar_images = find_similar_images(query_feature, top_k=8)
        
        # Prepare response
        response = {
            "success": True,
            "results": []
        }
        
        # For each similar image, read it and convert to base64
        for result in similar_images:
            img_path = result["path"]
            try:
                with open(img_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    response["results"].append({
                        "image_data": f"data:image/jpeg;base64,{img_data}",
                        "similarity": result["similarity"],
                        "path": os.path.basename(img_path)
                    })
            except Exception as e:
                print(f"Error processing result image {img_path}: {e}")
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Initialize the database at startup
if __name__ == "__main__":
    # image folder path
    IMAGE_FOLDER = "/home/parisol/Downloads/Img_annot/Img_annot/images/"
    initialize_database(IMAGE_FOLDER)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)