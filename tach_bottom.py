import cv2
import numpy as np
import os
import glob
from scipy import stats

# --- CẤU HÌNH ---
input_folder = "train/color_layers"
output_root = "train/patches"
os.makedirs(output_root, exist_ok=True)

blank_size = 32 
bg_color_value = 128 

image_files = glob.glob(os.path.join(input_folder, "**", "*bottom.png"), recursive=True)
print(f"Đang xử lý {len(image_files)} ảnh...")

for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None: continue
        
    filename = os.path.basename(img_path)
    base_name = filename.split('_')[0]
    
    save_dir = os.path.join(output_root, base_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_crop.png")

    # 1. Tạo Mask
    diff = cv2.absdiff(img, (bg_color_value, bg_color_value, bg_color_value))
    mask = np.any(diff > 10, axis=2).astype(np.uint8) # 0/1
    
    # 2. XÁC ĐỊNH CHIỀU NGANG (TRỤC X)
    v_proj = np.sum(mask, axis=0)
    non_zero_cols = v_proj[v_proj > 0]
    
    found_text = False
    
    if len(non_zero_cols) > 0:
        # Tự động tìm độ dày đường kẻ nhiễu (Mode)
        mode_res = stats.mode(non_zero_cols, keepdims=True)
        noise_level = mode_res.mode[0]
        
        # Lấy các cột có chiều cao > noise_level (Chứa thông tin chữ)
        valid_indices = np.where(v_proj > noise_level)[0]
        
        if len(valid_indices) > 0:
            x_start = valid_indices[0]
            x_end = valid_indices[-1]
            
            if (x_end - x_start) > 5: # Đủ rộng để là chữ
                # Padding X
                pad_x = 2
                h_img, w_img = img.shape[:2]
                x1 = max(0, x_start - pad_x)
                x2 = min(w_img, x_end + pad_x)
                
                # Cắt vùng ảnh theo chiều ngang (đã chứa chữ + nhiễu dọc)
                crop_x_mask = mask[:, x1:x2]
                crop_x_img = img[:, x1:x2]
                
                # 3. XÁC ĐỊNH CHIỀU DỌC (TRỤC Y) - LOẠI BỎ NHIỄU RỜI RẠC
                # Tính tổng pixel cho từng dòng (Horizontal Projection)
                h_proj = np.sum(crop_x_mask, axis=1)
                
                # Tìm các dòng có dữ liệu (non-zero rows)
                non_zero_rows = np.where(h_proj > 0)[0]
                
                if len(non_zero_rows) > 0:
                    # GOM NHÓM CÁC DÒNG LIỀN KỀ (TÌM ĐẢO)
                    # Cho phép đứt quãng nhỏ (gap <= 2px) vẫn tính là cùng 1 khối
                    diffs = np.diff(non_zero_rows)
                    split_indices = np.where(diffs > 2)[0] + 1 
                    groups = np.split(non_zero_rows, split_indices)
                    
                    # Tìm nhóm tốt nhất (Có tổng lượng pixel lớn nhất)
                    best_group = None
                    max_mass = -1
                    
                    for group in groups:
                        if len(group) == 0: continue
                        
                        # Tính "Trọng lượng" của nhóm = Tổng số pixel trong các dòng đó
                        # Chữ sẽ có trọng lượng lớn hơn hẳn so với vài đốm nhiễu
                        mass = np.sum(h_proj[group])
                        
                        if mass > max_mass:
                            max_mass = mass
                            best_group = group
                    
                    # Cắt theo nhóm tốt nhất tìm được
                    if best_group is not None:
                        found_text = True
                        y1 = best_group[0]
                        y2 = best_group[-1]
                        
                        # Padding Y
                        pad_y = 2
                        y1 = max(0, y1 - pad_y)
                        y2 = min(h_img, y2 + pad_y)
                        
                        final_crop = crop_x_img[y1:y2, :]
                        cv2.imwrite(out_path, final_crop)

    # Xử lý trường hợp không tìm thấy gì -> Ảnh Blank
    if not found_text:
        blank = np.full((blank_size, blank_size, 3), bg_color_value, dtype=np.uint8)
        cv2.imwrite(out_path, blank)
        # print(f"Blank: {filename}")

print("Hoàn tất.")
