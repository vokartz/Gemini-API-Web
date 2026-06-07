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

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from io import BytesIO

try:
    from ...utils import logger
except Exception:  # pragma: no cover - standalone script usage
    import logging

    logger = logging.getLogger("gemini_watermark_remover")

# Assets directory (same folder as this script)
ASSETS_DIR = Path(__file__).parent

# Watermark sizes to try. Only bg_48 and bg_96 are shipped as captures; the rest
# are derived from the nearest shipped capture via resampling so that detection
# can match watermarks rendered at non-standard scales.
CANDIDATE_SIZES = (32, 48, 64, 96, 128)
SHIPPED_SIZES = (48, 96)

# Minimum NCC score required to accept a match. Slightly below the original 0.75
# so real watermarks over busy/dark backgrounds aren't missed, but not so low
# that random bright corners are matched and then darkened by reverse blending.
SCORE_THRESHOLD = float(os.getenv("WATERMARK_SCORE_THRESHOLD", "0.68"))


def load_alpha_map(size):
    """Load or derive an alpha map for the given watermark size.

    Shipped sizes are read directly from ``bg_<size>.png``; other sizes are
    resampled from the nearest shipped capture so coverage isn't limited to the
    two bundled templates.
    """
    bg_path = ASSETS_DIR / f"bg_{size}.png"
    if bg_path.exists():
        bg_img = Image.open(bg_path).convert("RGB")
        bg_array = np.array(bg_img, dtype=np.float32)
        # Take max of RGB channels and normalize to [0, 1]
        return np.max(bg_array, axis=2) / 255.0

    # Derive from the nearest shipped capture.
    nearest = min(SHIPPED_SIZES, key=lambda s: abs(s - size))
    base = load_alpha_map(nearest)
    base_img = Image.fromarray((base * 255).astype(np.uint8))
    resized = base_img.resize((size, size), Image.Resampling.BILINEAR)
    return np.array(resized, dtype=np.float32) / 255.0


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

    Gemini renders its watermark in the bottom-right corner. We try several sizes
    and, for each, keep the best NCC peak among the top coarse candidates while
    biasing toward the bottom-right region, then refine to pixel accuracy.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    # Convert image to grayscale numpy array
    img_gray = img.convert("L")
    I = np.array(img_gray, dtype=np.float32) / 255.0
    width, height = img.size
    H_orig, W_orig = I.shape

    scale = 4
    W_s, H_s = width // scale, height // scale
    if W_s < 24 or H_s < 24:
        return None

    img_small = img_gray.resize((W_s, H_s), Image.Resampling.BILINEAR)
    I_small = np.array(img_small, dtype=np.float32) / 255.0

    best_match = None
    best_score = -1.0

    for size in CANDIDATE_SIZES:
        if size > H_orig or size > W_orig:
            continue
        try:
            alpha_map = get_alpha_map(size)
        except Exception as exc:
            logger.debug(f"Filigran alfa haritası yüklenemedi (boyut {size}): {exc}")
            continue

        # Downsample template
        alpha_img = Image.fromarray((alpha_map * 255).astype(np.uint8))
        size_s = max(1, size // scale)
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

        T_o_mean = alpha_map.mean()
        T_o_std = alpha_map.std()
        if T_o_std < 1e-5:
            T_o_std = 1.0
        T_o_norm = (alpha_map - T_o_mean) / (T_o_std * size * size)

        # Consider the top coarse candidates instead of only the global peak so a
        # genuine bottom-right watermark isn't lost to a higher-scoring false peak
        # elsewhere in the image.
        flat = ncc.ravel()
        top_k = min(8, flat.size)
        top_idx = np.argpartition(flat, -top_k)[-top_k:]
        candidates = [np.unravel_index(idx, ncc.shape) for idx in top_idx]

        for y_s, x_s in candidates:
            y_c = y_s * scale
            x_c = x_s * scale

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
                        score = float(np.sum(p_norm * T_o_norm))
                        if score > best_score:
                            best_score = score
                            best_match = {
                                "logo_size": size,
                                "y": y,
                                "x": x,
                                "score": score,
                            }

    if best_match:
        logger.debug(
            "Filigran adayı: boyut=%sx%s konum=(%s,%s) skor=%.4f eşik=%.2f"
            % (
                best_match["logo_size"],
                best_match["logo_size"],
                best_match["x"],
                best_match["y"],
                best_match["score"],
                SCORE_THRESHOLD,
            )
        )

    if best_match and best_match["score"] >= SCORE_THRESHOLD:
        return best_match
    return None


def bottom_right_fallback(width, height):
    """Fixed bottom-right region to use when template matching fails.

    Gemini's watermark sits in the bottom-right corner with a margin that scales
    with image size, so we can still attempt reverse alpha blending there.
    """
    config = detect_watermark_config(width, height)
    size = config["logo_size"]
    margin = config["margin"]
    if size > width or size > height:
        return None
    x = width - size - margin
    y = height - size - margin
    # Clamp inside the image in case the margin pushes us out of bounds.
    x = max(0, min(x, width - size))
    y = max(0, min(y, height - size))
    return {"logo_size": size, "x": x, "y": y, "score": 0.0, "fallback": True}


def detect_watermark_config(width, height):
    """
    Legacy helper to detect watermark configuration based on image dimensions.
    """
    if width > 1024 and height > 1024:
        return {"logo_size": 96, "margin": 64}
    else:
        return {"logo_size": 48, "margin": 32}


def remove_watermark(image, verbose=False, allow_fallback=False):
    """
    Remove Gemini watermark from image by auto-detecting its position and size
    using template matching, then applying reverse alpha blending.

    Args:
        image: PIL Image, file path, or bytes
        verbose: Print debug information
        allow_fallback: If detection fails, still attempt reverse alpha blending
            on the fixed bottom-right corner region where Gemini places its mark.

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
    if not match and allow_fallback:
        width, height = img.size
        match = bottom_right_fallback(width, height)
        if match:
            logger.debug(
                "Filigran tespit edilemedi; sağ-alt köşe fallback uygulanıyor "
                "(boyut=%sx%s konum=(%s,%s))."
                % (match["logo_size"], match["logo_size"], match["x"], match["y"])
            )
    if not match:
        logger.debug("Filigran tespit edilemedi; görsel değiştirilmeden döndürülüyor.")
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

    # Vectorized reverse alpha blending over the watermark region.
    alpha = np.clip(alpha_map.astype(np.float32), 0.0, MAX_ALPHA)
    mask = alpha >= ALPHA_THRESHOLD
    one_minus_alpha = 1.0 - alpha

    region = img_array[y : y + logo_size, x : x + logo_size, :]
    a3 = alpha[:, :, None]
    restored = (region - a3 * LOGO_VALUE) / one_minus_alpha[:, :, None]
    restored = np.clip(np.round(restored), 0, 255)
    region[mask] = restored[mask]
    img_array[y : y + logo_size, x : x + logo_size, :] = region

    if verbose:
        print(f"Pixels modified: {int(mask.sum())}")

    # Convert back to PIL Image
    return Image.fromarray(img_array.astype(np.uint8), 'RGB')


def remove_watermark_bytes(image_bytes, output_format='PNG', quality=95, allow_fallback=False):
    """
    Remove watermark and return as bytes.

    Args:
        image_bytes: Input image as bytes
        output_format: Output format (PNG, JPEG, etc.)
        quality: JPEG quality (1-100)
        allow_fallback: Blindly reverse-blend the bottom-right corner when template
            matching fails. Off by default — forcing it corrupts (blackens) the corner
            of images that don't actually carry the Gemini watermark.

    Returns:
        Processed image as bytes
    """
    result = remove_watermark(image_bytes, allow_fallback=allow_fallback)
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
