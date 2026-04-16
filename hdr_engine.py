import cv2
import numpy as np
from pathlib import Path
import argparse

def simple_white_balance(img):
    """
    Apply a simple gray world assumption white balance to remove harsh color casts
    """
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    # Mild correction instead of aggressive to keep some warmth
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def apply_clahe(img, clip_limit=1.5, tile_grid_size=(8,8)):
    """
    Enhance local contrast using CLAHE on the L-channel in LAB space
    This brings out details in shadows and window areas without washing out colors.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """
    Apply unsharp mask to crisp up real-estate textures (brick, wood, crisp lines)
    """
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(img - blurred) < threshold
        np.copyto(sharpened, img, where=low_contrast_mask)
    return sharpened

def process_bracketed_set(image_paths, output_path):
    print(f"Loading {len(image_paths)} images for HDR fusion...")
    images = []
    base_shape = None
    
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"Failed to read {p}")
            continue
            
        if base_shape is None:
            base_shape = img.shape[:2]
        else:
            # Force resize if slight mismatch (common in some drone shots/handhelds)
            if img.shape[:2] != base_shape:
                img = cv2.resize(img, (base_shape[1], base_shape[0]))
        images.append(img)
        
    if len(images) < 2:
        print("Need at least 2 valid images in the set for HDR fusion.")
        if len(images) == 1:
            print("Only 1 image found; skipping HDR, applying base enhancements directly.")
            res_mertens_8bit = images[0]
        else:
            return False
    else:
        print("Aligning images using AlignMTB to fix camera jitter...")
        aligner = cv2.createAlignMTB()
        aligner.process(images, images)

        print("Merging exposures using Mertens Fusion...")
        merge_mertens = cv2.createMergeMertens()
        res_mertens = merge_mertens.process(images)
        
        # Mertens outputs float32 in [0,1], scale back to uint8
        res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype(np.uint8)

    print("Applying Base Enhancements Layer...")
    # 1. White balance
    enhanced = simple_white_balance(res_mertens_8bit)
    
    # 2. Local contrast handling (CLAHE) - punchy but realistic
    enhanced = apply_clahe(enhanced, clip_limit=1.8, tile_grid_size=(8,8))
    
    # 3. Sharpening
    enhanced = unsharp_mask(enhanced, amount=0.6)

    print(f"Saving final output to {output_path}")
    cv2.imwrite(str(output_path), enhanced)
    return True

def main():
    parser = argparse.ArgumentParser(description="HDR Engine and Base Enhancement Layer for PixelDwell")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing a set of bracketed images (e.g., a tight cluster folder)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output HDR image (e.g., output/hdr_result.jpg)")
    
    args = parser.parse_args()
    input_p = Path(args.input_dir)
    exts = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for ext in exts:
        image_paths.extend(list(input_p.glob(ext)) + list(input_p.glob(ext.upper())))
        
    image_paths = sorted(image_paths)
    if not image_paths:
         print(f"No images found in input directory: {args.input_dir}")
         return
         
    # Pass all images in the folder to be fused
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    process_bracketed_set(image_paths, args.output)

if __name__ == "__main__":
    main()
