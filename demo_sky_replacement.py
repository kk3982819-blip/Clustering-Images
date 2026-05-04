import logging
from pathlib import Path
from sky_processor import SkyReplacementEngine
import cv2

def run_demo():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # 1. Setup Engine
    engine = SkyReplacementEngine(sky_assets_dir="static/sky_assets")
    
    # 2. Select an input image
    input_dir = Path("input")
    images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    if not images:
        logging.error("No images found in input/ directory. Please add an image to test.")
        return

    # Use the first image for the demo
    input_image = images[0]
    output_image = Path("output/demo_sunset_replacement.jpg")
    output_image.parent.mkdir(parents=True, exist_ok=True)
    
    logging.info("--- Starting Demo: Sunset Sky Replacement ---")
    logging.info("Input: %s", input_image)
    
    # 3. Process
    result = engine.process(input_image, sky_type="sunset", output_path=output_image)
    
    if result is not None:
        logging.info("--- Success! ---")
        logging.info("Final result saved to: %s", output_image)
    else:
        logging.error("Sky replacement failed.")

if __name__ == "__main__":
    run_demo()
