import tensorflow as tf

IMG_WIDTH = 512
IMG_HEIGHT = 512
NUM_CATEGORIES = 2

def conv_block(inputs, num_filters):
    x = tf.keras.Sequential([
        # Convolutional Layer
        tf.keras.layers.Conv2D(num_filters, 3, padding='same'),
        # Batch Normalization Layer
        tf.keras.layers.BatchNormalization(),
        # Activation Layer
        tf.keras.layers.Activation('relu'),
        # Convolutional Layer
        tf.keras.layers.Conv2D(num_filters, 3, padding='same'),
        # Batch Normalization Layer
        tf.keras.layers.BatchNormalization(),
        # Activation Layer
        tf.keras.layers.Activation('relu')
    ])(inputs)

    return x


# Defining the Unet++ Model
def unet_plus_plus_model(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), num_classes=2, deep_supervision=True):
	inputs = tf.keras.Input(shape=(512,512,3))
	pretrained_base = tf.keras.applications.EfficientNetB3(
        include_top=False,
        pooling='None',
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    )
	# Encoding Path
	x_00 = conv_block(inputs, 64)
	x_10 = conv_block(tf.keras.layers.MaxPooling2D()(x_00), 128)
	x_20 = conv_block(tf.keras.layers.MaxPooling2D()(x_10), 256)
	x_30 = conv_block(tf.keras.layers.MaxPooling2D()(x_20), 512)
	x_40 = conv_block(tf.keras.layers.MaxPooling2D()(x_30), 1024)

	# Nested Decoding Path
	x_01 = conv_block(tf.keras.layers.concatenate(
		[x_00, tf.keras.layers.UpSampling2D()(x_10)]), 64)
	x_11 = conv_block(tf.keras.layers.concatenate(
		[x_10, tf.keras.layers.UpSampling2D()(x_20)]), 128)
	x_21 = conv_block(tf.keras.layers.concatenate(
		[x_20, tf.keras.layers.UpSampling2D()(x_30)]), 256)
	x_31 = conv_block(tf.keras.layers.concatenate(
		[x_30, tf.keras.layers.UpSampling2D()(x_40)]), 512)

	x_02 = conv_block(tf.keras.layers.concatenate(
		[x_00, x_01, tf.keras.layers.UpSampling2D()(x_11)]), 64)
	x_12 = conv_block(tf.keras.layers.concatenate(
		[x_10, x_11, tf.keras.layers.UpSampling2D()(x_21)]), 128)
	x_22 = conv_block(tf.keras.layers.concatenate(
		[x_20, x_21, tf.keras.layers.UpSampling2D()(x_31)]), 256)

	x_03 = conv_block(tf.keras.layers.concatenate(
		[x_00, x_01, x_02, tf.keras.layers.UpSampling2D()(x_12)]), 64)
	x_13 = conv_block(tf.keras.layers.concatenate(
		[x_10, x_11, x_12, tf.keras.layers.UpSampling2D()(x_22)]), 128)

	x_04 = conv_block(tf.keras.layers.concatenate(
		[x_00, x_01, x_02, x_03, tf.keras.layers.UpSampling2D()(x_13)]), 64)

	# Deep Supervision Path
	# If deep supervision is enabled, then the model will output the segmentation maps
	# at each stage of the decoding path
	if deep_supervision:
		outputs = [
			tf.keras.layers.Conv2D(num_classes, 1)(x_01),
			tf.keras.layers.Conv2D(num_classes, 1)(x_02),
			tf.keras.layers.Conv2D(num_classes, 1)(x_03),
			tf.keras.layers.Conv2D(num_classes, 1)(x_04)
		]
		# Concatenating the segmentation maps
		outputs = tf.keras.layers.concatenate(outputs, axis=0)

	# If deep supervision is disabled, then the model will output the final segmentation map
	# which is the segmentation map at the end of the decoding path
	else:
		outputs = tf.keras.layers.Conv2D(num_classes, 1)(x_04)

	# Creating the model
	model = tf.keras.Model(
		inputs=inputs, outputs=outputs, name='Unet_plus_plus')

	# Returning the model
	return model


# Testing the model
if __name__ == "__main__":
	# Creating the model
	model = unet_plus_plus_model(input_shape=(
		300, 300, 3), num_classes=2, deep_supervision=True)

	# Printing the model summary
	model.summary()
