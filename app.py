import numpy as np
from PIL import Image, ImageDraw
import gradio as gr
import base64
from io import BytesIO

# Define style names
style_names = [
    "Watercolor",
    "Cyberpunk",
    "Anime",
    "Oil Painting",
    "Sketch"
]

def create_watercolor_demo():
    """Create a watercolor-style landscape"""
    img = Image.new('RGB', (200, 200), (173, 216, 230))  # Light blue sky
    draw = ImageDraw.Draw(img)
    
    # Mountains with yellow highlights
    draw.polygon([(0, 100), (80, 40), (160, 90), (200, 60), (200, 200), (0, 200)], 
                fill=(100, 120, 110))  # Mountain base
    draw.polygon([(20, 80), (60, 50), (100, 75)], 
                fill=(255, 255, 0))  # Yellow mountain highlight
    
    # Yellow sun
    draw.ellipse([(150, 20), (190, 60)], fill=(255, 255, 0))
    
    return img

def create_cyberpunk_demo():
    """Create a cyberpunk city scene"""
    img = Image.new('RGB', (200, 200), (0, 0, 40))  # Dark blue night sky
    draw = ImageDraw.Draw(img)
    
    # Buildings with yellow neon
    for x in range(0, 200, 40):
        height = np.random.randint(80, 180)
        draw.rectangle([(x, height), (x+30, 200)], fill=(40, 40, 60))
        # Yellow neon signs
        draw.rectangle([(x+5, height+20), (x+25, height+30)], fill=(255, 255, 0))
    
    return img

def create_anime_demo():
    """Create an anime-style character"""
    img = Image.new('RGB', (200, 200), (255, 182, 193))  # Pink background
    draw = ImageDraw.Draw(img)
    
    # Stylized character with yellow elements
    # Hair
    draw.ellipse([(50, 30), (150, 130)], fill=(255, 255, 0))  # Yellow hair
    # Face
    draw.ellipse([(70, 50), (130, 110)], fill=(255, 220, 200))
    
    return img

def create_oil_painting_demo():
    """Create an oil painting style still life"""
    img = Image.new('RGB', (200, 200), (139, 69, 19))  # Brown background
    draw = ImageDraw.Draw(img)
    
    # Yellow vase
    draw.ellipse([(60, 60), (140, 160)], fill=(255, 255, 0))
    # Yellow flowers
    for i in range(3):
        x = 70 + i * 30
        draw.ellipse([(x, 40), (x+20, 60)], fill=(255, 255, 0))
    
    return img

def create_sketch_demo():
    """Create a sketch-style drawing"""
    img = Image.new('RGB', (200, 200), (255, 255, 255))  # White background
    draw = ImageDraw.Draw(img)
    
    # Sketch lines in dark gray
    draw.line([(50, 50), (150, 50)], fill=(100, 100, 100), width=2)
    # Yellow highlighting
    draw.rectangle([(60, 60), (140, 140)], fill=(255, 255, 0), outline=(100, 100, 100))
    
    return img

# Style to function mapping
style_creators = {
    "Watercolor": create_watercolor_demo,
    "Cyberpunk": create_cyberpunk_demo,
    "Anime": create_anime_demo,
    "Oil Painting": create_oil_painting_demo,
    "Sketch": create_sketch_demo
}

# Define color loss function
def yellow_loss(image, strength=0.8):
    """Reduces yellow colors in the image."""
    try:
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Extract RGB channels
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        
        # Define yellow as high red and green, low blue
        yellow_mask = np.logical_and(np.logical_and(r > 0.5, g > 0.5), b < 0.4)
        
        # Apply transformation
        if np.any(yellow_mask):
            r[yellow_mask] *= (1 - strength * 0.7)
            g[yellow_mask] *= (1 - strength)
            b[yellow_mask] += (1 - b[yellow_mask]) * strength
            
            # Update the image array
            img_array[:, :, 0] = r
            img_array[:, :, 1] = g
            img_array[:, :, 2] = b
        
        # Convert back to PIL image
        return Image.fromarray((img_array * 255).astype(np.uint8))
    except Exception as e:
        print(f"Error in yellow_loss: {e}")
        # Return original image if processing fails
        return image

# Function to apply color loss
def apply_color_loss(style, strength, image_input=None):
    """Apply yellow loss to an image."""
    try:
        # If user uploaded an image, use it
        if image_input is not None:
            # Resize to reasonable dimensions for speed
            try:
                # Handle different input types
                if isinstance(image_input, np.ndarray):
                    image = Image.fromarray(image_input)
                else:
                    image = image_input
                
                # Resize for speed but keep aspect ratio
                image.thumbnail((300, 300), Image.LANCZOS)
            except Exception as e:
                print(f"Error processing input image: {e}")
                # Use demo image as fallback
                image = style_creators[style]()
        else:
            # Create a styled demo image
            image = style_creators[style]()
        
        # Apply yellow loss
        result = yellow_loss(image, strength)
        
        # Create a side-by-side comparison
        comparison = Image.new('RGB', (image.width * 2 + 10, image.height), (240, 240, 240))
        comparison.paste(image, (0, 0))
        comparison.paste(result, (image.width + 10, 0))
        
        # Add labels
        draw = ImageDraw.Draw(comparison)
        draw.text((10, 10), f"Original ({style})", fill=(0, 0, 0))
        draw.text((image.width + 20, 10), f"Yellow Loss: {strength:.1f}", fill=(0, 0, 0))
        
        return comparison
    
    except Exception as e:
        print(f"Error in apply_color_loss: {e}")
        # Return a simple fallback image
        return style_creators["Watercolor"]()

# Create minimal Gradio interface
demo = gr.Interface(
    fn=apply_color_loss,
    inputs=[
        gr.Dropdown(choices=style_names, value=style_names[0], label="Style"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.8, step=0.1, label="Yellow Loss Strength"),
        gr.Image(label="Upload an image (optional)", type="pil")
    ],
    outputs=gr.Image(label="Result (Before and After)"),
    title="Yellow Loss Demo",
    description="This demo shows how yellow loss affects different artistic styles. Each style includes yellow elements that will be modified by the effect.",
    examples=[
        ["Watercolor", 0.8, None],
        ["Cyberpunk", 0.5, None],
        ["Anime", 0.9, None],
        ["Oil Painting", 0.7, None],
        ["Sketch", 0.6, None]
    ],
    cache_examples=True  # Cache examples for faster loading
)

# Launch the app
if __name__ == "__main__":
    demo.launch() 