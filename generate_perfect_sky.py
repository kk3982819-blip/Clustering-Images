import cv2
import numpy as np

def create_perfect_sky():
    # Create a 4K sky asset to ensure high resolution
    w, h = 3840, 2160
    sky = np.zeros((h, w, 3), dtype=np.float32)

    # Gradient colors (BGR)
    # Top: Deep blue/purple
    color_top = np.array([180, 100, 80], dtype=np.float32)
    # Mid-top: Magenta/Purple transition
    color_mid1 = np.array([120, 70, 140], dtype=np.float32)
    # Mid-bottom: Vibrant Orange
    color_mid2 = np.array([50, 120, 255], dtype=np.float32)
    # Horizon: Bright yellow/gold
    color_bottom = np.array([40, 200, 255], dtype=np.float32)

    # Create vertical gradient
    for y in range(h):
        t = y / h
        if t < 0.3:
            # Top to mid1
            local_t = t / 0.3
            color = color_top * (1 - local_t) + color_mid1 * local_t
        elif t < 0.6:
            # mid1 to mid2
            local_t = (t - 0.3) / 0.3
            color = color_mid1 * (1 - local_t) + color_mid2 * local_t
        else:
            # mid2 to bottom
            local_t = (t - 0.6) / 0.4
            color = color_mid2 * (1 - local_t) + color_bottom * local_t
        
        sky[y, :] = color

    # Draw the sun
    sun_x = int(w * 0.55)  # Slightly to the right
    sun_y = int(h * 0.75)  # Near the horizon

    # 1. Huge faint orange glow
    glow1_radius = int(w * 0.4)
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_sq = (x_grid - sun_x)**2 + (y_grid - sun_y)**2
    
    glow1_mask = np.exp(-dist_sq / (glow1_radius**2 * 0.5))
    glow_color = np.array([0, 100, 255], dtype=np.float32)  # Orange
    for c in range(3):
        sky[:, :, c] += glow1_mask * glow_color[c] * 0.6

    # 2. Medium yellow glow
    glow2_radius = int(w * 0.15)
    glow2_mask = np.exp(-dist_sq / (glow2_radius**2 * 0.2))
    glow2_color = np.array([0, 220, 255], dtype=np.float32)  # Yellow
    for c in range(3):
        sky[:, :, c] += glow2_mask * glow2_color[c] * 0.8

    # 3. Sharp bright sun core
    sun_core_radius = int(w * 0.015)
    core_mask = dist_sq < sun_core_radius**2
    # Anti-aliased edge for core
    dist_sqrt = np.sqrt(dist_sq)
    core_blend = np.clip((sun_core_radius + 5 - dist_sqrt) / 5.0, 0, 1)
    
    sky[:, :, 0] = np.where(core_blend > 0, sky[:, :, 0] * (1 - core_blend) + 255 * core_blend, sky[:, :, 0])
    sky[:, :, 1] = np.where(core_blend > 0, sky[:, :, 1] * (1 - core_blend) + 255 * core_blend, sky[:, :, 1])
    sky[:, :, 2] = np.where(core_blend > 0, sky[:, :, 2] * (1 - core_blend) + 255 * core_blend, sky[:, :, 2])

    # Final clip and save
    sky = np.clip(sky, 0, 255).astype(np.uint8)
    
    # Save as clear_day.jpg to map to "sunny"
    cv2.imwrite("static/sky_assets/clear_day.jpg", sky)
    # Also save as golden_hour.jpg just in case
    cv2.imwrite("static/sky_assets/golden_hour.jpg", sky)
    print("Generated perfect sky assets.")

if __name__ == "__main__":
    create_perfect_sky()
