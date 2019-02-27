import cv2

cap = cv2.VideoCapture('test_video.mp4')
currentFrame = 0
while True:
	# Capture frame-by-frame
	ret, frame = cap.read()
	if not ret:
		break
	# Handles the mirroring of the current frame
	# frame = cv2.flip(frame, 1)
	frame = cv2.resize(frame, (640, 360))
	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Saves image of the current frame in jpg file
	# name = 'frame' + str(currentFrame) + '.jpg'
	# cv2.imwrite(name, frame)

	# Display the resulting frame
	cv2.imshow('frame', gray)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	# To stop duplicate images
	currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
