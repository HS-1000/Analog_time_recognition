import tensorflow as tf
from tensorflow.keras import layers

INPUT_SHAPE = (256, 256, 3)
MODEL_PATH = "clock.keras"

def conv_dw_block(x, pointwise_filters, stride):
	""" Depthwise Separable Convolution Block (MobileNet 스타일) """
	x = layers.DepthwiseConv2D(
		kernel_size = 3, 
		strides = stride, 
		padding = "same"
	)(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)

	x = layers.Conv2D(
		pointwise_filters, 
		kernel_size = 1, 
		strides = 1,
		padding = "same"
	)(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	return x

if __name__ == "__main__":
	inputs = layers.Input(shape=INPUT_SHAPE)
	x = layers.Conv2D(32, (3, 3), strides=2, padding="same")(inputs)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)

	x = conv_dw_block(x, 64, stride=1)
	x = conv_dw_block(x, 128, stride=2)
	x = conv_dw_block(x, 256, stride=1)
	x = conv_dw_block(x, 256, stride=2)
	x = conv_dw_block(x, 512, stride=1)
	x = conv_dw_block(x, 512, stride=2)

	x = layers.GlobalAveragePooling2D()(x)
	x = layers.Dense(256)(x)
	x = layers.Dropout(0.1)(x)
	x = layers.Dense(64, activation="relu")(x)
	outputs = layers.Dense(2, activation='linear')(x)

	model = tf.keras.models.Model(inputs, outputs)
	model.summary()
	model.save(MODEL_PATH)


