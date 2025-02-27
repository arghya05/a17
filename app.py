import numpy as np
from PIL import Image
import gradio as gr
import os
import gc

# Define style names
style_names = [
    "Watercolor",
    "Cyberpunk",
    "Anime",
    "Oil Painting",
    "Sketch"
]

# Define color loss functions
def yellow_loss(image, strength=0.8):
    """Reduces yellow colors in the generated images."""
    # Convert PIL image to numpy array
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Extract RGB channels
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]
    
    # Define yellow as high red and green, low blue
    yellow_mask = ((r > 0.5) & (g > 0.5) & (b < 0.4))
    
    # Transform yellow areas toward blue/purple
    r[yellow_mask] = r[yellow_mask] * (1 - strength * 0.7)  # Reduce red somewhat
    g[yellow_mask] = g[yellow_mask] * (1 - strength)        # Reduce green more
    b[yellow_mask] = b[yellow_mask] + (1 - b[yellow_mask]) * strength  # Increase blue
    
    # Update the image
    img_array[:, :, 0] = r
    img_array[:, :, 1] = g
    img_array[:, :, 2] = b
    
    # Convert back to PIL image
    return Image.fromarray((img_array * 255).astype(np.uint8))

def cyan_loss(image, strength=0.8):
    """Reduces cyan colors in the generated images."""
    # Convert PIL image to numpy array
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Extract RGB channels
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]
    
    # Define cyan as low red, high green and blue
    cyan_mask = ((r < 0.4) & (g > 0.5) & (b > 0.5))
    
    # Transform cyan areas toward red/orange
    r[cyan_mask] = r[cyan_mask] + (1 - r[cyan_mask]) * strength
    g[cyan_mask] = g[cyan_mask] * (1 - strength * 0.7)
    b[cyan_mask] = b[cyan_mask] * (1 - strength * 0.7)
    
    # Update the image
    img_array[:, :, 0] = r
    img_array[:, :, 1] = g
    img_array[:, :, 2] = b
    
    # Convert back to PIL image
    return Image.fromarray((img_array * 255).astype(np.uint8))

def magenta_loss(image, strength=0.8):
    """Reduces magenta colors in the generated images."""
    # Convert PIL image to numpy array
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Extract RGB channels
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]
    
    # Define magenta as high red and blue, low green
    magenta_mask = ((r > 0.5) & (g < 0.4) & (b > 0.5))
    
    # Transform magenta areas toward green
    r[magenta_mask] = r[magenta_mask] * (1 - strength)
    g[magenta_mask] = g[magenta_mask] + (1 - g[magenta_mask]) * strength
    b[magenta_mask] = b[magenta_mask] * (1 - strength)
    
    # Update the image
    img_array[:, :, 0] = r
    img_array[:, :, 1] = g
    img_array[:, :, 2] = b
    
    # Convert back to PIL image
    return Image.fromarray((img_array * 255).astype(np.uint8))

# Color loss functions dictionary
color_loss_functions = {
    "None": lambda img, s: img,
    "Yellow Loss": yellow_loss,
    "Cyan Loss": cyan_loss,
    "Magenta Loss": magenta_loss
}

# Function to generate images
def generate_image(prompt, style, color_loss_type, seed, strength, image_input=None):
    # For demo purposes, we'll use a placeholder image if no diffusers
    if image_input is None:
        # Create a simple colored image as placeholder
        if style == "Watercolor":
            color = (173, 216, 230)  # Light blue
        elif style == "Cyberpunk":
            color = (255, 105, 180)  # Hot pink
        elif style == "Anime":
            color = (255, 165, 0)    # Orange
        elif style == "Oil Painting":
            color = (139, 69, 19)    # Saddle brown
        else:  # Sketch
            color = (169, 169, 169)  # Dark gray
            
        # Create a gradient image
        width, height = 512, 512
        image = Image.new('RGB', (width, height), color)
        
        # Add some text to show the prompt
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(image)
            try:
                # Try several common fonts
                for font_name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"]:
                    try:
                        font = ImageFont.truetype(font_name, 20)
                        break
                    except:
                        continue
                else:  # If no font was found
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # Wrap text
            words = prompt.split()
            lines = []
            current_line = []
            for word in words:
                if len(' '.join(current_line + [word])) <= 40:
                    current_line.append(word)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            
            text = '\n'.join(lines)
            
            # Add style info
            text += f"\n\nStyle: {style}"
            if seed >= 0:
                text += f"\nSeed: {seed}"
            
            # Draw text with outline for better visibility
            x, y = 20, 20
            # Draw outline
            for offset_x, offset_y in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                draw.multiline_text((x+offset_x, y+offset_y), text, fill=(0,0,0), font=font)
            # Draw text
            draw.multiline_text((x, y), text, fill=(255,255,255), font=font)
            
        except Exception as e:
            print(f"Error adding text to image: {e}")
    else:
        image = image_input
    
    # Apply color loss if selected
    if color_loss_type != "None" and color_loss_functions[color_loss_type] is not None:
        try:
            print(f"Applying {color_loss_type}...")
            image = color_loss_functions[color_loss_type](image, strength)
        except Exception as e:
            print(f"Error applying color loss: {e}")
            # Return original image if color loss fails
    
    return image, seed if seed >= 0 else 42

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Color Loss Demo") as demo:
        gr.Markdown("# Color Loss Demo")
        gr.Markdown("This demo shows how different color loss functions can modify images.")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="a serene landscape with mountains and a lake")
                style = gr.Dropdown(label="Style", choices=style_names, value=style_names[0])
                color_loss = gr.Dropdown(
                    label="Color Loss Type", 
                    choices=list(color_loss_functions.keys()), 
                    value="None"
                )
                seed = gr.Number(label="Seed (use -1 for random)", value=42, precision=0)
                strength = gr.Slider(label="Color Loss Strength", minimum=0.1, maximum=1.0, value=0.8, step=0.1)
                image_input = gr.Image(label="Upload an image (optional)", type="pil")
                generate_btn = gr.Button("Apply Color Loss")
            
            with gr.Column():
                output_image = gr.Image(label="Result Image")
                used_seed = gr.Number(label="Seed Used", precision=0)
        
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, style, color_loss, seed, strength, image_input],
            outputs=[output_image, used_seed]
        )
        
        gr.Markdown("## Examples")
        gr.Examples(
            examples=[
                ["a serene landscape with mountains and a lake", "Watercolor", "None", 42, 0.8, None],
                ["a serene landscape with mountains and a lake", "Cyberpunk", "Yellow Loss", 123, 0.8, None],
                ["a serene landscape with mountains and a lake", "Anime", "Cyan Loss", 456, 0.8, None],
                ["a futuristic cityscape at night", "Oil Painting", "Magenta Loss", 789, 0.8, None],
                ["a portrait of a person", "Sketch", "Yellow Loss", 1024, 0.8, None],
            ],
            inputs=[prompt, style, color_loss, seed, strength, image_input],
            outputs=[output_image, used_seed],
        )
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch() 