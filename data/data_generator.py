import os
import cv2
import numpy as np
import random

# 폴더 설정
input_folder = r'C:\\mmp/data/origin'
output_folder = r'C:\\mmp/data/augmented'
os.makedirs(output_folder, exist_ok=True)

def augment_image(img):
	h, w = img.shape[:2]

	# 회전
	angle = random.uniform(-10, 10)
	M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
	img = cv2.warpAffine(img, M_rot, (w, h), borderMode=cv2.BORDER_REFLECT)

	# 스케일 조절
	scale = random.uniform(0.9, 1.1)
	img = cv2.resize(img, None, fx=scale, fy=scale)
	img = cv2.resize(img, (w, h))

	# 이동
	tx = random.randint(-int(w * 0.05), int(w * 0.05))
	ty = random.randint(-int(h * 0.05), int(h * 0.05))
	M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
	img = cv2.warpAffine(img, M_trans, (w, h), borderMode=cv2.BORDER_REFLECT)

	# 밝기 조절
	brightness = random.uniform(0.7, 1.3)
	img = np.clip(img * brightness, 0, 255).astype(np.uint8)

	return img

def process_images(input_folder, output_folder, num_augments=5):
	for filename in os.listdir(input_folder):
		if filename.endswith('.jpg') or filename.endswith('.png'):
			filepath = os.path.join(input_folder, filename)
			img = cv2.imread(filepath)

			if img is None:
				print(f"이미지 로드 실패: {filename}")
				continue

			y_value = filename[-8:-4]  # 예: "1234"
			origin_id = filename[:-8]

			for i in range(num_augments):
				aug_img = augment_image(img)
				save_name = f"{origin_id}_aug{i}_{y_value}.jpg"
				save_path = os.path.join(output_folder, save_name)
				cv2.imwrite(save_path, aug_img)

				print(f"저장됨: {save_name}")

# 실행
if __name__ == "__main__":
	process_images(input_folder, output_folder, num_augments=3)
