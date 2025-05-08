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
import pickle

# Import ViT model and utils 
from timm.models.vision_transformer import VisionTransformer
import torch.nn as nn

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Custom ViT wrapper class for SatMAE
class VitExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # Forward pass through the ViT model
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        if hasattr(self.model, 'pos_embed'):
            x = x + self.model.pos_embed
            
        x = self.model.pos_drop(x)
        
        # Pass through transformer blocks
        for blk in self.model.blocks:
            x = blk(x)
            
        x = self.model.norm(x)
        
        # Return all tokens including CLS token
        return {"last_hidden_state": x}

# Load SatMAE ViT model
def load_model(checkpoint_path, large=True):
    
    # Define ViT-Large model configuration
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=1024,  # ViT-Large
        depth=24,        # ViT-Large
        num_heads=16,    # ViT-Large
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm
    )
    
    if not large:
        # Define ViT-Base model configuration
        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=768,    # Changed from 1024 to 768 for ViT-Base
            depth=12,         # Changed from 24 to 12 for ViT-Base
            num_heads=12,     # Changed from 16 to 12 for ViT-Base
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=nn.LayerNorm
        )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove prefix if needed (depends on how model was saved)
    for key in list(state_dict.keys()):
        if key.startswith('module.'):
            state_dict[key[7:]] = state_dict[key]
            del state_dict[key]
    
    # Load weights
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint with message: {msg}")
    
    # Wrap in our extractor class
    return VitExtractor(model)

# Define path to pretrained model
checkpoint_path = "./models/fmow_pretrain.pth"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(checkpoint_path, large=True).to(device)
model.eval()  # Set to evaluation mode

# Setup image transformations
transform = T.Compose([
    T.Resize((224, 224)),  # Resize image to ViT input size
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
def extract_features_batch(images, batch_size=256):
    """
    Extract features from a batch of images using SatMAE ViT.
    
    Args:
        images: List of PIL images
        batch_size: Number of images to process in each batch
        
    Returns:
        numpy array of features (num_images x feature_dim)
    """
    num_images = len(images)
    all_features = []
    
    # Process in batches
    for i in range(0, num_images, batch_size):
        batch_images = images[i:i+batch_size]
        batch_tensors = []
        
        # Convert each image to tensor
        for img in batch_images:
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Check if this is a small region and enhance if needed
            width, height = img.size
            if width < 100 or height < 100:
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.3)  # Slightly boost contrast
            
            # Apply transformations
            tensor = transform(img)
            batch_tensors.append(tensor)
        
        # Stack tensors into a batch
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        # Extract features using SatMAE ViT
        with torch.no_grad():
            outputs = model(batch_tensor)
            # Use CLS token features (first token)
            batch_features = outputs["last_hidden_state"][:, 0].cpu().numpy().astype(np.float32)
            
        all_features.append(batch_features)
    
    # Concatenate all batches
    return np.vstack(all_features)

def initialize_database(image_folders, scale_factors=[1.0, 0.66, 0.33], batch_size=256, db_filename="satmae_database.pkl"):
    """
    Initialize the image database with an image pyramid approach using batched processing.
    If the database file exists, load from it instead of creating a new one.
    """

    # Check if database file exists
    if os.path.exists(db_filename):
        print(f"Loading image database from '{db_filename}'...")
        with open(db_filename, 'rb') as f:
            image_db = pickle.load(f)

        # Rebuild FAISS index
        features_array = image_db["features"]
        d = features_array.shape[1]
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(features_array)
        index.add(features_array)
        image_db["index"] = index

        return image_db

    print("No saved database found. Initializing new database...")

    image_paths = []
    for image_folder in image_folders: 
        print(f"Building image database from {image_folder}...")
        image_paths += list(Path(image_folder).glob("**/*.jpg"))
        image_paths += list(Path(image_folder).glob("**/*.png"))

    batch_images, batch_paths, batch_metadata = [], [], []
    features, paths, metadata = [], [], []
    img_count = 0

    for img_idx, img_path in enumerate(image_paths):
        try:
            image = Image.open(img_path).convert("RGB")
            orig_width, orig_height = image.size
            
            for scale in scale_factors:
                if scale != 1.0:
                    new_width, new_height = int(orig_width * scale), int(orig_height * scale)
                    if new_width < 64 or new_height < 64:
                        continue
                    scaled_image = image.resize((new_width, new_height))
                else:
                    scaled_image = image

                batch_images.append(scaled_image)
                batch_paths.append(str(img_path))
                batch_metadata.append({
                    "original_path": str(img_path),
                    "scale": scale,
                    "width": scaled_image.width,
                    "height": scaled_image.height
                })

                if len(batch_images) >= batch_size:
                    try:
                        batch_features = extract_features_batch(batch_images, batch_size)
                        features.extend(batch_features)
                        paths.extend(batch_paths)
                        metadata.extend(batch_metadata)
                        batch_images, batch_paths, batch_metadata = [], [], []
                        img_count += batch_size
                        if img_count % 1000 == 0:
                            print(f"Processed {img_count} images")
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        batch_images, batch_paths, batch_metadata = [], [], []

            if (img_idx + 1) % 100 == 0:
                print(f"Processed {img_idx + 1}/{len(image_paths)} images ({len(features) + len(batch_images)} total entries)")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    if batch_images:
        try:
            batch_features = extract_features_batch(batch_images, len(batch_images))
            features.extend(batch_features)
            paths.extend(batch_paths)
            metadata.extend(batch_metadata)
        except Exception as e:
            print(f"Error processing final batch: {e}")

    if not features:
        print("No features extracted! Check your image folder and image formats.")
        return

    features_array = np.array(features).astype('float32')
    d = features_array.shape[1]
    faiss.normalize_L2(features_array)
    index = faiss.IndexFlatIP(d)
    index.add(features_array)

    image_db = {
        "features": features_array,
        "paths": paths,
        "metadata": metadata,
        "index": index
    }

    # Save to file
    print(f"Saving database to '{db_filename}'...")
    image_db_to_save = image_db.copy()
    image_db_to_save["index"] = None  # FAISS index cannot be pickled directly
    with open(db_filename, 'wb') as f:
        pickle.dump(image_db_to_save, f)

    print(f"Database initialized with {len(paths)} entries from {len(image_paths)} images")
    return image_db
    """
    Initialize the image database with an image pyramid approach using batched processing.
    """
    
    image_paths = []
    for image_folder in image_folders: 
        print(f"Building image database from {image_folder}...")

        image_paths += (list(Path(image_folder).glob("**/*.jpg")) + list(Path(image_folder).glob("**/*.png")))  

    # Lists to collect data for batch processing
    batch_images = []
    batch_paths = []
    batch_metadata = []
    
    features = []
    paths = []
    metadata = []
    
    # Process images
    img_count = 0
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
                
                # Add to batch
                batch_images.append(scaled_image)
                batch_paths.append(str(img_path))
                batch_metadata.append({
                    "original_path": str(img_path),
                    "scale": scale,
                    "width": scaled_image.width,
                    "height": scaled_image.height
                })
                
                # Process batch when it reaches batch_size
                if len(batch_images) >= batch_size:
                    try:
                        batch_features = extract_features_batch(batch_images, batch_size)
                        features.extend(batch_features)
                        paths.extend(batch_paths)
                        metadata.extend(batch_metadata)
                        
                        # Reset batch
                        batch_images = []
                        batch_paths = []
                        batch_metadata = []
                        
                        img_count += batch_size
                        if img_count % 1000 == 0:
                            print(f"Processed {img_count} images")
                            
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        # Reset batch on error
                        batch_images = []
                        batch_paths = []
                        batch_metadata = []
            
            if (img_idx + 1) % 100 == 0:
                print(f"Processed {img_idx + 1}/{len(image_paths)} images ({len(features) + len(batch_images)} total entries)")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Process final batch if any
    if batch_images:
        try:
            batch_features = extract_features_batch(batch_images, len(batch_images))
            features.extend(batch_features)
            paths.extend(batch_paths)
            metadata.extend(batch_metadata)
        except Exception as e:
            print(f"Error processing final batch: {e}")
    
    if not features:
        print("No features extracted! Check your image folder and image formats.")
        return
    
    # Convert to numpy array
    features_array = np.array(features).astype('float32')
    
    # Create FAISS index for fast similarity search
    d = features_array.shape[1]  # Dimension of feature vectors
    index = faiss.IndexFlatIP(d)  # Inner product similarity
    
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
    Extract features from an image using SatMAE ViT.
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
    
    # Extract features using SatMAE ViT
    with torch.no_grad():
        outputs = model(image_tensor)
        # Use CLS token features (first token)
        feature = outputs["last_hidden_state"][:, 0].cpu().numpy().astype(np.float32)
    
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
            response["results"].append({
                "similarity": result["adjusted_similarity"],
                "raw_similarity": result["raw_similarity"],
                "size_ratio": result["size_ratio"],
                "path": os.path.basename(img_path)
            })

        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in process_region: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Initialize the database at startup
if __name__ == "__main__":
    # image folder path
    IMAGE_FOLDERS = ["/home/pari/Desktop/Img_annot/Img_annot/images/", "/home/pari/Desktop/Img_annot/Img_annot/annotations/" ]
    image_db = initialize_database(IMAGE_FOLDERS)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', use_reloader=False, port=5000)
