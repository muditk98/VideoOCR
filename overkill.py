import argparse
import cv2
import numpy as np
import pytesseract
from imutils.object_detection import non_max_suppression
from matplotlib import pyplot as plt
from pytesseract import Output


def thressher(orig):
	img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

	threshes = [
		cv2.threshold(cv2.GaussianBlur(img, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
		cv2.threshold(cv2.GaussianBlur(img, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
		cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
		cv2.threshold(cv2.medianBlur(img, 5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
		cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
		cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
		cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
		# cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],
		# cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1],
		# cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1],
		# cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1]
	]

	titles = [
		'gaus9+otsu',
		'guas7+otsu',
		'guas5+otsu',
		'median5+otsu',
		'median3+otsu',
		'gaus5+adagaus 31 2',
		'gaus3+adagaus 31 2',
	]
	return threshes


def tess(image):
	d = pytesseract.image_to_data(image, output_type=Output.DICT, lang=args['lang'])
	image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	n_boxes = len(d['level'])
	for i in range(n_boxes):
		if int(d['conf'][i]) < 80 or not d['text'][i] or d['text'][i].isspace():
			continue
		(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.putText(image, d['text'][i], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
		# cv2.putText(image, translator.translate(d['text'][i]).text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
		print(str(i) + ':' + d['text'][i])
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def decode_predictions(scores, geometry):
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	for y in range(numRows):
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		for x in range(numCols):
			if scoresData[x] < args["min_confidence"]:
				continue

			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	return rects, confidences


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to input image")
ap.add_argument("-east", "--east", type=str, help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.7,help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320, help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type=int, default=320, help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type=float, default=0.0, help="amount of padding to add to each border of ROI")
ap.add_argument("-l", "--lang", type=str, help="Language", default='eng')
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
orig = image.copy()
(origH, origW) = image.shape[:2]

(newW, newH) = (32 * (origW // 32), 32 * (origH // 32))

rW = origW / float(newW)
rH = origH / float(newH)

image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"
]

# print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)

results = []

for (startX, startY, endX, endY) in boxes:
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	dX = int((endX - startX) * args["padding"])
	dY = int((endY - startY) * args["padding"])

	startX = max(0, startX - dX)
	startY = max(0, startY - dY)
	endX = min(origW, endX + (dX * 2))
	endY = min(origH, endY + (dY * 2))

	roi = orig[startY:endY, startX:endX]
	th_imgs = thressher(roi)
	for i in range(len(th_imgs)):
		img = tess(th_imgs[i])
		# plt.subplot(3,4,i+1)
		plt.imshow(img)
		plt.show()
		# plt.title(titles[i])
		# plt.xticks([]), plt.yticks([])
		print('-' * 10)

	# config = "--oem 1 --psm 7"
	# text = pytesseract.image_to_string(roi, config=config, lang=args['lang'])
	# results.append(((startX, startY, endX, endY), text))

# results = sorted(results, key=lambda r: r[0][1])
#
# for ((startX, startY, endX, endY), text) in results:
# 	print("OCR TEXT")
# 	print("========")
# 	print("{}\n".format(text))
#
# 	text = "".join(text).strip()
# 	output = orig.copy()
# 	cv2.rectangle(output, (startX, startY), (endX, endY),
# 				  (0, 0, 255), 2)
# 	cv2.putText(output, text, (startX, startY - 20),
# 				cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 3)
#
# 	cv2.imshow("Text Detection", cv2.resize(output, (720, 480)))
# 	cv2.waitKey(0)
