import os
import argparse
import logging
from pathlib import Path

# Note: In a real system, these would be imported directly if they were modules,
# or run via subprocess/Celery tasks. We import their main logic functions here.
from hdr_engine import process_bracketed_set
from api_orchestrator import EnhancementOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [PIPELINE] - %(message)s")

def run_end_to_end_pipeline(input_dir, output_dir, ai_features):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting PixelDwell Pipeline on input directory: {input_dir}")
    
    # --- PHASE 1: Clustering (Mocked / Assumed Complete) ---
    # In reality, you'd call cluster_images.py here to group the raw uploads into "viewpoints".
    # For this pipeline, we assume the input_dir ALREADY represents a clustered viewpoint 
    # (a set of bracketed images of the same exact room angle).
    logging.info("Phase 1: Scene Clustering complete (assuming input_dir is a clustered viewpoint).")
    
    images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpeg"))
    if not images:
        logging.error(f"No images found in {input_dir}")
        return False
        
    # --- PHASE 2 & 3: HDR Fusion and Base Enhancements ---
    logging.info("Phase 2 & 3: Starting HDR Fusion and Base Local Enhancements.")
    hdr_output_path = output_dir / "base_hdr_enhanced.jpg"
    
    # Process the bracketed set locally using OpenCV
    success = process_bracketed_set(images, hdr_output_path)
    
    if not success or not hdr_output_path.exists():
        logging.error("HDR Base Enhancements failed. Aborting pipeline.")
        return False
        
    logging.info(f"HDR + Base Enhancement complete: {hdr_output_path}")
    
    # --- PHASE 4 & 5: AI Orchestration (External API Calls) ---
    if ai_features:
        logging.info(f"Phase 4 & 5: Routing to AI Orchestrator for features: {ai_features}")
        orchestrator = EnhancementOrchestrator()
        
        # The Orchestrator takes the base enhanced image and applies AI features sequentially
        final_image_path = orchestrator.process_image(
            image_path=hdr_output_path,
            requested_features=ai_features,
            output_dir=output_dir / "ai_results"
        )
        logging.info(f"AI Enhancements complete. Final output generated at: {final_image_path}")
    else:
        logging.info("No external AI features requested. Pipeline finished at Base Enhancements.")
        
    logging.info("PixelDwell Pipeline Execution Complete! 🎉")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full PixelDwell Pipeline")
    parser.add_argument("--input", required=True, help="Path to input directory containing a bracketed set of photos")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--ai-features", nargs="*", default=["sky_replacement"], help="List of AI features to run (e.g., sky_replacement virtual_staging)")
    
    args = parser.parse_args()
    run_end_to_end_pipeline(args.input, args.output, args.ai_features)
