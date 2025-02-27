---
title: Color Loss Demo
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 3.32.0
app_file: app.py
pinned: false
---

# Color Loss Demo

This app demonstrates different color loss functions that can modify the color distribution in images.

## Features

- Apply different color loss effects:
  - Yellow Loss: Reduces yellow tones in the image
  - Cyan Loss: Reduces cyan tones in the image
  - Magenta Loss: Reduces magenta tones in the image
- Control the strength of the color loss effect
- Upload your own images to apply color loss effects
- Generate placeholder images in different styles

## How to Use

1. Enter a prompt (for placeholder image text)
2. Select a style for the placeholder image
3. Choose a color loss type (or "None" for no effect)
4. Set a seed value (for reproducibility)
5. Adjust the color loss strength
6. Upload your own image (optional)
7. Click "Apply Color Loss"

## Examples

The app includes several example configurations that you can try with a single click.

## Technical Details

This app uses:
- PIL for image processing
- NumPy for efficient array operations
- Custom color loss functions that target specific color ranges 