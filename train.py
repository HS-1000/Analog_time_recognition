import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import data_tools 

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
DATA_DIRS = ["data/origin", "data/augmented"]

train_x, train_y, val_x, val_y = data_tools.load_dataset(DATA_DIRS, img_size=IMG_SIZE)

model = tf.keras.models.load_model("./clock.keras")
model.compile(
	optimizer = "adam",
	loss = "mse",
	metrics = ["mae"]
)

checkpoint_cb = ModelCheckpoint(
	"checkpoint.keras",
	save_best_only = True,
	monitor = "val_loss"
)

model.fit(
	train_x, train_y,
	epochs = 200,
	batch_size = BATCH_SIZE,
	validation_data = (val_x, val_y),
	callbacks = [checkpoint_cb]
)

