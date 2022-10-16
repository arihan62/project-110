# To Capture Frame
import cv2

# To process image array
import numpy as np

# To Load the Pre-trained Model
import tensorflow as tf

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Loading the pre-trained model : keras_model.h5
mymodel = tf.keras.models.load_model('keras_model.h5')


while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		
		frame = cv2.flip(frame , 1)

		
		resized_frame = cv2.resize(frame , (224,224))

		
		resized_frame = np.expand_dims(resized_frame , axis = 0)

		
		resized_frame = resized_frame / 255

		# Getting predictions from the model
		predictions = mymodel.predict(resized_frame)

		# Converting the data in the array to percentage confidence 
		rock = int(predictions[0][0]*100)
		paper = int(predictions[0][1]*100)
		scissor = int(predictions[0][2]*100)

		
		print(f"Rock: {rock} %, Paper: {paper} %, Scissor: {scissor} %")

		
		cv2.imshow('feed' , frame)

		
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break


camera.release()


cv2.destroyAllWindows()