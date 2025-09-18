import os
from pathlib import Path
from PIL import Image
import numpy as np

def to_rgb_mask(img_path, out_path, threshold=0, use_otsu=False):
    """Convert a single image to an RGB black/white mask and save as JPG."""
    img = Image.open(img_path).convert("L")  # force grayscale
    arr = np.array(img, dtype=np.uint8)

    if use_otsu:
        # Simple Otsu threshold
        hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
        p = hist / hist.sum()
        omega = np.cumsum(p)
        mu = np.cumsum(p * np.arange(256))
        mu_t = mu[-1]
        sigma_b2 = (mu_t * omega - mu)**2 / (omega * (1 - omega) + 1e-12)
        t = np.nanargmax(sigma_b2)
        bin_mask = (arr > t).astype(np.uint8) * 255
    else:
        if threshold > 0:
            bin_mask = (arr > threshold).astype(np.uint8) * 255
        else:
            bin_mask = (arr > 0).astype(np.uint8) * 255

    # Stack to RGB
    rgb_mask = np.repeat(bin_mask[:, :, None], 3, axis=2)

    # Save as JPG (always truecolor, no palette)
    Image.fromarray(rgb_mask, mode="RGB").save(out_path, format="JPEG", quality=95)

def process_folder(input_folder, output_folder, threshold=0, use_otsu=False):
    """Process all images in a folder and save converted masks in output_folder as JPGs."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    supported_exts = [".png", ".jpg", ".jpeg", ".tif", ".bmp"]

    for img_file in input_folder.iterdir():
        if img_file.suffix.lower() in supported_exts:
            # Change extension to .jpg
            out_name = img_file.stem + ".jpg"
            out_path = output_folder / out_name
            print(f"Converting {img_file} -> {out_path}")
            to_rgb_mask(img_file, out_path, threshold=threshold, use_otsu=use_otsu)

if __name__ == "__main__":

    input_masks = "C:\\Users\\user\\Documents\\Reassembly2d_Sources\\Pictures\\Masks\\mask_carli"
    output_masks = "C:\\Users\\user\\Documents\\Reassembly2d_Sources\\Pictures\\Masks\\mask_carli_rgb_jpg"

    # Example usage
    process_folder(input_masks, output_masks, threshold=0)