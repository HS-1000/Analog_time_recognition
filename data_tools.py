import cv2
import numpy as np
import os

def parse_label_from_filename(filenames):
	"""
	Args: <str>, <list>, <np.ndarray>
	Return: shape=(N, 2) np.ndarray (cos, sin)
	"""
	def label_from_one(fname):
		hh = int(fname[-8:-6])
		mm = int(fname[-6:-4])
		angle_deg = (hh % 12) * 30 + mm * 0.5
		angle_rad = np.deg2rad(angle_deg)
		return [np.cos(angle_rad), np.sin(angle_rad)]

	if isinstance(filenames, str):
		return np.array(label_from_one(filenames), dtype=np.float32)

	elif isinstance(filenames, (list, np.ndarray)):
		return np.array([label_from_one(fname) for fname in filenames], dtype=np.float32)

	else:
		raise TypeError("입력은 str, list[str], np.ndarray[str] 중 하나여야 합니다.")

def preprocess_image(path, img_size=(256, 256)):
	""" 히스토그램, 라플라시안, canny 3채널 """
	img = cv2.imread(path)
	img = cv2.resize(img, img_size)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	eq = cv2.equalizeHist(gray)
	canny = cv2.Canny(eq, 100, 200)
	lap = cv2.Laplacian(eq, cv2.CV_64F)
	lap = cv2.convertScaleAbs(lap)
	stacked = np.stack([eq, canny, lap], axis=-1).astype(np.float32) / 255.0
	return stacked

def load_dataset(data_dirs, val_index=420, img_size=(256, 256)):
	train_x, train_y, val_x, val_y = [], [], [], []

	for folder in data_dirs:
		for fname in os.listdir(folder):
			if not fname.endswith('.jpg'):
				continue
			path = os.path.join(folder, fname)
			img = preprocess_image(path, img_size=img_size)
			label = parse_label_from_filename(fname)
			id_str = fname[:3]

			val_index = str(val_index)
			val_index = "0" * (3 - len(val_index)) + val_index
			if id_str < val_index:
				train_x.append(img)
				train_y.append(label)
			else:
				val_x.append(img)
				val_y.append(label)

	return (
		np.array(train_x), np.array(train_y),
		np.array(val_x), np.array(val_y)
	)
