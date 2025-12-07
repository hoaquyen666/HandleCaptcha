import cv2
import numpy as np
import os
import glob


input_folder = "captcha-image"
output_root = "train/color_layers"
os.makedirs(output_root, exist_ok=True)

image_files = glob.glob(os.path.join(input_folder, "*.png")) + \
              glob.glob(os.path.join(input_folder, "*.jpg")) + \
              glob.glob(os.path.join(input_folder, "*.jpeg"))

if len(image_files) == 0:
    input_folder = "captcha_goc"
    image_files = glob.glob(os.path.join(input_folder, "*.png")) + \
                  glob.glob(os.path.join(input_folder, "*.jpg")) + \
                  glob.glob(os.path.join(input_folder, "*.jpeg"))

print(f"Found {len(image_files)} image files in {input_folder}")

bg_color = np.array([108, 109, 103], dtype=np.float32)

for image_path in image_files:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image: {image_path}")
        continue

    filename = os.path.basename(image_path)
    base_name, ext = os.path.splitext(filename)

    diff = np.abs(img.astype(np.float32) - bg_color)
    distance = np.sqrt(np.sum(diff ** 2, axis=2))
    bg_mask = distance < 20

    normalized = img.copy()
    normalized[bg_mask] = [128, 128, 128]

    h_img, w_img = normalized.shape[:2]
    if h_img > 32:
        y1, y2 = 32, min(64, h_img)
        normalized[y1:y2, :, :] = [128, 128, 128]

    if h_img > 118:
        yb = 118
        normalized[yb:h_img, :, :] = [128, 128, 128]

    out_base_dir = os.path.join(output_root, base_name)
    os.makedirs(out_base_dir, exist_ok=True)

    diff_from_gray = np.abs(normalized.astype(np.float32) - np.array([128, 128, 128], dtype=np.float32))
    distance_from_gray = np.sqrt(np.sum(diff_from_gray ** 2, axis=2))
    colored_mask = distance_from_gray > 5

    if not np.any(colored_mask):
        print(f"No colored pixels found in {filename}")
        continue

    colored_pixels = normalized[colored_mask]
    unique_colors, counts = np.unique(colored_pixels.reshape(-1, 3), axis=0, return_counts=True)

    min_pixels_per_color = 30
    for idx, (color, count) in enumerate(zip(unique_colors, counts)):
        if count < min_pixels_per_color:
            continue

        layer = np.full_like(normalized, [128, 128, 128])

        diff_c = np.abs(normalized.astype(np.int16) - color.astype(np.int16))
        dist_c = np.sqrt(np.sum(diff_c ** 2, axis=2))
        mask_c = dist_c < 5

        layer[mask_c] = normalized[mask_c]

        # Split layer into top (0-32) and bottom (62-118) regions
        h_layer, w_layer = layer.shape[:2]

        top_start, top_end = 0, min(32, h_layer)
        bottom_start, bottom_end = min(62, h_layer), min(118, h_layer)

        if top_end > top_start:
            top_region = layer[top_start:top_end, :, :]
            out_top_path = os.path.join(out_base_dir, f"{base_name}_layer_{idx}_top{ext}")
            cv2.imwrite(out_top_path, top_region)

        if bottom_end > bottom_start:
            bottom_region = layer[bottom_start:bottom_end, :, :]
            out_bottom_path = os.path.join(out_base_dir, f"{base_name}_layer_{idx}_bottom{ext}")
            cv2.imwrite(out_bottom_path, bottom_region)

    print(f"Processed: {filename}")

print("Completed splitting color layers.")
