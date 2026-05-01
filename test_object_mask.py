import logging
from pathlib import Path
from mask_generator import generate_object_mask

def test_object_mask():
    logging.basicConfig(level=logging.INFO)
    input_img = Path(r"uploads_temp/raw/DSC04225_HDR.jpg")
    output_mask = Path("output_test_sky/object_mask_test.png")
    
    # Simulate a click on an object (e.g., a chair or cable)
    # These coordinates are just for testing the SAM flow
    points = [[500, 500]] 
    
    print(f"Testing object mask generation for {input_img}...")
    success = generate_object_mask(input_img, output_mask, points)
    
    if success:
        print(f"SUCCESS: Mask saved to {output_mask}")
    else:
        print("FAILED: Mask generation failed.")

if __name__ == "__main__":
    test_object_mask()
