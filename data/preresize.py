import os
import cv2
from tqdm import tqdm

INPUT_DIRS = ["data/origin", "data/augmented"]
OUTPUT_DIR = "data/light"
TARGET_SIZE = (256, 256)

# 출력 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

def resize_and_save_all():
    for input_dir in INPUT_DIRS:
        for fname in tqdm(os.listdir(input_dir), desc=f"Processing {input_dir}"):
            if not fname.endswith(".jpg"):
                continue

            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(OUTPUT_DIR, fname)

            try:
                img = cv2.imread(input_path)
                if img is None:
                    print(f"Failed to load {input_path}")
                    continue

                resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                cv2.imwrite(output_path, resized)
            except Exception as e:
                print(f"Error processing {fname}: {e}")

resize_and_save_all()
