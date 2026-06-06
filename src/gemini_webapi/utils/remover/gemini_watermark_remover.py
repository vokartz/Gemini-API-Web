#!/usr/bin/env python3
"""
Gemini Watermark Remover - Python Implementation

A Python port of the reverse alpha blending algorithm for removing
Gemini AI watermarks from generated images.

Based on: https://github.com/journey-ad/gemini-watermark-remover

Usage:
    python gemini_watermark_remover.py <image_path> [output_path]

Requirements:
    pip install pillow numpy
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
from io import BytesIO

# Assets directory (same folder as this script)
ASSETS_DIR = Path(__file__).parent


def load_alpha_map(size):
    """Load and calculate alpha map from watermark capture image."""
    bg_path = ASSETS_DIR / f"bg_{size}.png"
    if not bg_path.exists():
        raise FileNotFoundError(f"Alpha map file not found: {bg_path}")

    bg_img = Image.open(bg_path).convert('RGB')
    bg_array = np.array(bg_img, dtype=np.float32)

    # Take max of RGB channels and normalize to [0, 1]
    alpha_map = np.max(bg_array, axis=2) / 255.0
    return alpha_map


# Pre-computed alpha maps (lazy loaded)
_ALPHA_MAPS = {}


def get_alpha_map(size):
    """Get cached alpha map for given size."""
    if size not in _ALPHA_MAPS:
        _ALPHA_MAPS[size] = load_alpha_map(size)
    return _ALPHA_MAPS[size]


def detect_watermark(img, verbose=False):
    """
    Detect the watermark size, position, and score using coarse-to-fine NCC template matching.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    
    # Convert image to grayscale numpy array
    img_gray = img.convert('L')
    I = np.array(img_gray, dtype=np.float32) / 255.0
    width, height = img.size
    
    scale = 4
    W_s, H_s = width // scale, height // scale
    if W_s < 24 or H_s < 24:
        return None
        
    img_small = img_gray.resize((W_s, H_s), Image.Resampling.BILINEAR)
    I_small = np.array(img_small, dtype=np.float32) / 255.0
    
    best_match = None
    best_score = -1.0
    
    for size in [48, 96]:
        try:
            alpha_map = get_alpha_map(size)
        except Exception as exc:
            if verbose:
                print(f"Failed to load alpha map for size {size}: {exc}")
            continue
            
        # Downsample template
        alpha_img = Image.fromarray((alpha_map * 255).astype(np.uint8))
        size_s = size // scale
        alpha_small_img = alpha_img.resize((size_s, size_s), Image.Resampling.BILINEAR)
        T_small = np.array(alpha_small_img, dtype=np.float32) / 255.0
        
        h, w = T_small.shape
        if H_s < h or W_s < w:
            continue
            
        # Coarse template matching
        T_mean = T_small.mean()
        T_std = T_small.std()
        if T_std < 1e-5:
            T_std = 1.0
        T_norm = (T_small - T_mean) / (T_std * h * w)
        
        windows = sliding_window_view(I_small, (h, w))
        W_mean = windows.mean(axis=(2, 3), keepdims=True)
        W_std = windows.std(axis=(2, 3), keepdims=True)
        W_std = np.where(W_std < 1e-5, 1.0, W_std)
        
        W_norm = (windows - W_mean) / W_std
        ncc = np.sum(W_norm * T_norm, axis=(2, 3))
        
        y_s, x_s = np.unravel_index(np.argmax(ncc), ncc.shape)
        
        # Scale back to original resolution and refine
        y_c = y_s * scale
        x_c = x_s * scale
        
        best_fine_score = -1.0
        best_fine_pos = (0, 0)
        
        T_o_mean = alpha_map.mean()
        T_o_std = alpha_map.std()
        if T_o_std < 1e-5:
            T_o_std = 1.0
        T_o_norm = (alpha_map - T_o_mean) / (T_o_std * size * size)
        
        H_orig, W_orig = I.shape
        
        # Search window of ±8 pixels around coarse candidate
        for dy in range(-8, 9):
            for dx in range(-8, 9):
                y = y_c + dy
                x = x_c + dx
                if 0 <= y <= H_orig - size and 0 <= x <= W_orig - size:
                    patch = I[y : y + size, x : x + size]
                    p_mean = patch.mean()
                    p_std = patch.std()
                    if p_std < 1e-5:
                        p_std = 1.0
                    p_norm = (patch - p_mean) / p_std
                    score = np.sum(p_norm * T_o_norm)
                    if score > best_fine_score:
                        best_fine_score = score
                        best_fine_pos = (y, x)
                        
        if verbose:
            print(f"Size {size} best fine score: {best_fine_score} at {best_fine_pos}")
            
        if best_fine_score > best_score:
            best_score = best_fine_score
            best_match = {
                "logo_size": size,
                "y": best_fine_pos[0],
                "x": best_fine_pos[1],
                "score": best_fine_score
            }
            
    if best_match and best_match["score"] >= 0.75:
        return best_match
    return None


def detect_watermark_config(width, height):
    """
    Legacy helper to detect watermark configuration based on image dimensions.
    """
    if width > 1024 and height > 1024:
        return {"logo_size": 96, "margin": 64}
    else:
        return {"logo_size": 48, "margin": 32}


def remove_watermark(image, verbose=False):
    """
    Remove Gemini watermark from image by auto-detecting its position and size
    using template matching, then applying reverse alpha blending.

    Args:
        image: PIL Image, file path, or bytes
        verbose: Print debug information

    Returns:
        PIL Image with watermark removed (or original image if not detected)
    """
    # Handle different input types
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert('RGB')
    elif isinstance(image, bytes):
        img = Image.open(BytesIO(image)).convert('RGB')
    elif isinstance(image, Image.Image):
        img = image.convert('RGB')
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    match = detect_watermark(img, verbose=verbose)
    if not match:
        if verbose:
            print("Watermark not detected, returning image unchanged.")
        return img

    logo_size = match["logo_size"]
    y = match["y"]
    x = match["x"]
    score = match["score"]

    if verbose:
        print(f"Detected watermark size: {logo_size}x{logo_size} at position ({x}, {y}) with score {score:.4f}")

    # Load alpha map
    alpha_map = get_alpha_map(logo_size)

    # Convert to numpy for processing
    img_array = np.array(img, dtype=np.float32)

    # Constants
    ALPHA_THRESHOLD = 0.002  # Ignore noise
    MAX_ALPHA = 0.99         # Avoid division by zero
    LOGO_VALUE = 255.0       # White watermark

    pixels_modified = 0

    # Process watermark region
    for row in range(logo_size):
        for col in range(logo_size):
            alpha = alpha_map[row, col]

            # Skip very small alpha values (noise)
            if alpha < ALPHA_THRESHOLD:
                continue

            # Limit alpha to avoid division by near-zero
            alpha = min(alpha, MAX_ALPHA)
            one_minus_alpha = 1.0 - alpha

            # Apply reverse alpha blending to each RGB channel
            for c in range(3):
                watermarked = img_array[y + row, x + col, c]
                original = (watermarked - alpha * LOGO_VALUE) / one_minus_alpha
                img_array[y + row, x + col, c] = max(0, min(255, round(original)))

            pixels_modified += 1

    if verbose:
        print(f"Pixels modified: {pixels_modified}")

    # Convert back to PIL Image
    return Image.fromarray(img_array.astype(np.uint8), 'RGB')


def remove_watermark_bytes(image_bytes, output_format='PNG', quality=95):
    """
    Remove watermark and return as bytes.

    Args:
        image_bytes: Input image as bytes
        output_format: Output format (PNG, JPEG, etc.)
        quality: JPEG quality (1-100)

    Returns:
        Processed image as bytes
    """
    result = remove_watermark(image_bytes)
    output = BytesIO()

    save_kwargs = {"format": output_format}
    if output_format.upper() == "JPEG":
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True

    result.save(output, **save_kwargs)
    return output.getvalue()


def main():
    """Command-line interface."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
    else:
        output_path = image_path.parent / f"{image_path.stem}_clean{image_path.suffix}"

    if not image_path.exists():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    print(f"Processing: {image_path}")

    result = remove_watermark(image_path, verbose=True)
    result.save(output_path)

    print(f"Saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
