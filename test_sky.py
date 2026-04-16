import os
from pathlib import Path
from api_orchestrator import EnhancementOrchestrator

def main():
    # Use the CLEAN original (no annotations) for sky replacement
    target_img = Path(r"c:\Users\Mohammed kaif M\OneDrive\Desktop\clustering images\uploads_temp\raw\DSC04222_HDR.jpg")
    output_dir = Path(r"c:\Users\Mohammed kaif M\OneDrive\Desktop\clustering images\output_test_sky")

    orchestrator = EnhancementOrchestrator()
    print("Running orchestrator...")
    final = orchestrator.process_image(
        target_img,
        requested_features=["sky_replacement"],
        output_dir=str(output_dir)
    )
    print(f"Finished! Output saved to: {final}")

if __name__ == "__main__":
    main()
