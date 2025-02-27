import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from pathlib import Path
import random
import gc

# Set device - updated for Apple Silicon support
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"  # Use Metal Performance Shaders on Apple Silicon
else:
    device = "cpu"
print(f"Using device: {device}")

# Define paths to the learned embeddings
embedding_paths = [
    "learned_embeds (1).bin",
    "learned_embeds (2).bin",
    "learned_embeds (3).bin",
    "learned_embeds (4).bin",
    "learned_embeds (5).bin"
]

# Define style names (you can customize these based on what the embeddings represent)
style_names = [
    "Watercolor",
    "Cyberpunk",
    "Anime",
    "Oil Painting",
    "Sketch"
]

# Create a mapping of style names to embedding paths
style_to_embedding = {name: path for name, path in zip(style_names, embedding_paths)}

# Define the base prompt to use with all styles
base_prompt = "a serene landscape with mountains and a lake"

# Set different seeds for each style
seeds = {
    "Watercolor": 42,
    "Cyberpunk": 123,
    "Anime": 456,
    "Oil Painting": 789,
    "Sketch": 1024
}

# Function to load the model with a specific embedding
def load_model_with_embedding(embedding_path, token_name="<concept>"):
    # Load the base model
    # For MPS, we need to use float32 as float16 has limited support
    use_fp16 = device == "cuda"
    
    # Use a smaller model for faster generation
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",  # Smaller than v1-5
        torch_dtype=torch.float16 if use_fp16 else torch.float32
    ).to(device)
    
    # Disable safety checker for speed
    pipe.safety_checker = None
    
    # Load the learned embedding
    learned_embeds = torch.load(embedding_path)
    
    # Add the embedding to the tokenizer and text encoder
    token = token_name
    pipe.tokenizer.add_tokens(token)
    token_id = pipe.tokenizer.convert_tokens_to_ids(token)
    
    # Resize the token embeddings
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    
    # Get the embedding dimension from the model
    embedding_dim = pipe.text_encoder.get_input_embeddings().weight.shape[1]
    
    # Get the actual embedding tensor from the loaded file
    # The key might vary depending on how the embedding was saved
    embedding_tensor = None
    for key in learned_embeds:
        if isinstance(learned_embeds[key], torch.Tensor) and learned_embeds[key].shape[-1] == embedding_dim:
            embedding_tensor = learned_embeds[key]
            break
    
    if embedding_tensor is None:
        # If we couldn't find the embedding tensor by shape, try common keys
        common_keys = ["string_to_param", "emb_params", "embedding"]
        for key in common_keys:
            if key in learned_embeds:
                embedding_tensor = learned_embeds[key]
                break
    
    if embedding_tensor is None:
        # If still not found, use the first tensor in the dictionary
        embedding_tensor = next(iter(learned_embeds.values()))
    
    # Handle dimension mismatch
    if embedding_tensor.shape[0] != embedding_dim:
        print(f"Warning: Embedding dimension mismatch. Found {embedding_tensor.shape[0]}, expected {embedding_dim}.")
        print("Attempting to resize the embedding...")
        
        # Try to reshape if it's a multiple (e.g., 768 -> 1024)
        if embedding_tensor.shape[0] < embedding_dim:
            # Pad with zeros
            padded_tensor = torch.zeros(embedding_dim)
            padded_tensor[:embedding_tensor.shape[0]] = embedding_tensor
            embedding_tensor = padded_tensor
        else:
            # Truncate
            embedding_tensor = embedding_tensor[:embedding_dim]
        
        print(f"Resized embedding to {embedding_tensor.shape[0]}")
    
    # Set the embedding
    pipe.text_encoder.get_input_embeddings().weight.data[token_id] = embedding_tensor
    
    return pipe, token

# Define a yellow loss function
def yellow_loss(images, strength=0.8):
    """
    Implements a 'yellow_loss' function that penalizes yellow colors in the generated images.
    Yellow is characterized by high red and green with low blue.
    
    Args:
        images: Tensor of shape [batch_size, channels, height, width]
        strength: Strength of the loss effect (higher = stronger effect)
    
    Returns:
        Modified images with reduced yellow components
    """
    # Convert images to numpy for easier manipulation
    images_np = images.cpu().numpy()
    
    for i in range(images_np.shape[0]):
        # Extract RGB channels
        r = images_np[i, 0, :, :]
        g = images_np[i, 1, :, :]
        b = images_np[i, 2, :, :]
        
        # Define yellow as high red and green, low blue
        yellow_mask = ((r > 0.5) & (g > 0.5) & (b < 0.4))
        
        # Transform yellow areas toward blue/purple
        r[yellow_mask] = r[yellow_mask] * (1 - strength * 0.7)  # Reduce red somewhat
        g[yellow_mask] = g[yellow_mask] * (1 - strength)        # Reduce green more
        b[yellow_mask] = b[yellow_mask] + (1 - b[yellow_mask]) * strength  # Increase blue
        
        # Update the image
        images_np[i, 0, :, :] = r
        images_np[i, 1, :, :] = g
        images_np[i, 2, :, :] = b
    
    # Convert back to tensor
    return torch.from_numpy(images_np).to(images.device)

# Function to generate images with and without yellow_loss
def generate_images(style_name, embedding_path, base_prompt, seed):
    # Load model with the specific embedding
    pipe, token = load_model_with_embedding(embedding_path)
    
    # Enable memory efficient attention for better performance on M2
    pipe.enable_attention_slicing(slice_size=1)
    
    # Drastically reduce image size for faster generation
    height = 256  # Reduced from 384
    width = 256   # Reduced from 384
    
    # Reduce inference steps drastically for faster generation
    num_inference_steps = 15  # Reduced from 20
    
    # Set the seed for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Create the full prompt with the style token
    full_prompt = f"{base_prompt}, {token} style"
    
    # Generate image without yellow_loss
    print(f"Generating image with {style_name} style...")
    image_without_loss = pipe(
        full_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        height=height,
        width=width
    ).images[0]
    
    # Reset generator to the same seed for fair comparison
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Generate image with yellow_loss
    print(f"Generating image with {style_name} style and yellow_loss...")
    
    # Since callback isn't supported in your version, we'll generate the image first
    # and then apply the yellow_loss as a post-processing step
    image_with_loss_raw = pipe(
        full_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        height=height,
        width=width
    ).images[0]
    
    # Convert PIL image to tensor
    image_tensor = torch.from_numpy(np.array(image_with_loss_raw)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # Apply yellow loss
    modified_tensor = yellow_loss(image_tensor, strength=0.9)
    
    # Convert back to PIL image
    modified_array = (modified_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    image_with_loss = Image.fromarray(modified_array)
    
    # Create a debug image showing which areas were modified
    debug_image = image_tensor.clone()
    # Extract RGB channels
    r = debug_image[0, 0, :, :]
    g = debug_image[0, 1, :, :]
    b = debug_image[0, 2, :, :]
    # Identify yellow areas
    yellow_mask = ((r > 0.5) & (g > 0.5) & (b < 0.4))
    # Highlight yellow areas in magenta for visibility
    debug_tensor = debug_image.clone()
    debug_tensor[0, 0, yellow_mask] = 1  # Red channel
    debug_tensor[0, 1, yellow_mask] = 0  # Green channel
    debug_tensor[0, 2, yellow_mask] = 1  # Blue channel
    
    # Convert tensor to PIL image before saving
    debug_array = (debug_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    debug_image_pil = Image.fromarray(debug_array)
    # Save debug image
    debug_image_pil.save(output_dir / f"{style_name}_yellow_debug.png")
    
    # Clear memory
    try:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Try to clear MPS cache if available
        import gc
        gc.collect()
    except Exception as e:
        print(f"Warning: Could not clear cache: {e}")
    
    return image_without_loss, image_with_loss

# Create output directory
output_dir = Path("textual_inversion_results")

# Generate and save images for each style
results = {}

# Process one style at a time instead of all at once
for style_name, embedding_path in style_to_embedding.items():
    print(f"\nProcessing {style_name} style...")
    
    # Generate images
    image_without_loss, image_with_loss = generate_images(
        style_name, 
        embedding_path, 
        base_prompt, 
        seeds[style_name]
    )
    
    # Save images immediately
    output_dir.mkdir(exist_ok=True)
    image_without_loss.save(output_dir / f"{style_name}_original.png")
    image_with_loss.save(output_dir / f"{style_name}_yellow_loss.png")
    
    # Store results for display
    results[style_name] = {
        "original": image_without_loss,
        "yellow_loss": image_with_loss,
        "seed": seeds[style_name]
    }
    
    # Clear memory between styles
    try:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    except Exception as e:
        print(f"Warning: Could not clear cache: {e}")
    
    # Optional: Display progress
    print(f"Completed {style_name}. Images saved to {output_dir}")

# Display results
plt.figure(figsize=(20, 15))

for i, (style_name, images) in enumerate(results.items()):
    # Original image
    plt.subplot(5, 2, i*2 + 1)
    plt.imshow(np.array(images["original"]))
    plt.title(f"{style_name} (Seed: {images['seed']})")
    plt.axis('off')
    
    # Image with yellow_loss
    plt.subplot(5, 2, i*2 + 2)
    plt.imshow(np.array(images["yellow_loss"]))
    plt.title(f"{style_name} with Yellow Loss")
    plt.axis('off')

plt.tight_layout()
plt.savefig(output_dir / "comparison.png")
plt.show()

print(f"All images saved to {output_dir.absolute()}")
print("Seeds used:")
for style, seed in seeds.items():
    print(f"  {style}: {seed}") 