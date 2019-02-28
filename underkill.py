import argparse
import cv2
import numpy as np
import pytesseract
from imutils.object_detection import non_max_suppression
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to input image")
ap.add_argument("-east", "--east", default='frozen_east_text_detection.pb', type=str, help="path for input east")
ap.add_argument("-c", "--min-confidence", type=float, default=0.75, help="minimum confidence")
ap.add_argument("-w", "--width", type=int, default=320, help="nearest multiple of 32 for re-sized width")
ap.add_argument("-e", "--height", type=int, default=320, help="nearest multiple of 32 for re-sized height")
ap.add_argument("-p", "--padding", type=float, default=0.2, help="amount of padding to add to each border of ROI")
ap.add_argument("-l", "--lang", type=str, help="Language", default='eng')
ap.add_argument("-t", "--thresh", type=float, help="Threshold", default=0.3)
args = vars(ap.parse_args())


def decode_predictions(scores, geometry):
	(num_rows, num_cols) = scores.shape[2:4]
	rectangles = []
	confidences = []

	for y in range(num_rows):
		scores_data = scores[0, 0, y]
		x_data0 = geometry[0, 0, y]
		x_data1 = geometry[0, 1, y]
		x_data2 = geometry[0, 2, y]
		x_data3 = geometry[0, 3, y]
		angles_data = geometry[0, 4, y]

		for x in range(num_cols):
			if scores_data[x] < args["min_confidence"]:
				continue

			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			angle = angles_data[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			h = x_data0[x] + x_data2[x]
			w = x_data1[x] + x_data3[x]

			end_x = int(offsetX + (cos * x_data1[x]) + (sin * x_data2[x]))
			end_y = int(offsetY - (sin * x_data1[x]) + (cos * x_data2[x]))
			start_x = int(end_x - w)
			start_y = int(end_y - h)

			rectangles.append((start_x, start_y, end_x, end_y))
			confidences.append(scores_data[x])

	return rectangles, confidences


def apply_filters(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.GaussianBlur(img, (3, 3), 0)
	# img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 2)
	img = cv2.medianBlur(img, 3)
	img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
	# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	return img


def tess(image):
	config = "--oem 3 --psm 7"
	d = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang=args['lang'], config=config)
	n_boxes = len(d['level'])
	result = ''
	confidences = []
	for i in range(n_boxes):
		d['text'][i] = d['text'][i].replace('|', '')
		if int(d['conf'][i])/100.0 < args['min_confidence'] or not d['text'][i] or d['text'][i].isspace():
			continue
		# (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
		result += ' ' + d['text'][i]
		confidences.append(d['conf'][i])
		print(str(i) + ': ' + d['text'][i] + ' conf: ' + str(d['conf'][i]))
	# return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	if len(confidences):
		# print('here ' + result)
		return result.strip(), np.mean(confidences)
	else:
		return '', 0


def recognize(image):
	(origH, origW) = image.shape[:2]
	(newW, newH) = (32 * (origW // 32), 32 * (origH // 32))
	r_w = origW / float(newW)
	r_h = origH / float(newH)
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	# image = apply_filters(image)
	# image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	# plt.imshow(image)
	# plt.show()

	layer_names = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"
	]

	# print("[INFO] loading EAST text detector...")
	net = cv2.dnn.readNet(args["east"])

	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layer_names)

	(rectangles, confidences) = decode_predictions(scores, geometry)
	# print('before nms', len(rectangles))
	boxes = non_max_suppression(np.array(rectangles), probs=confidences, overlapThresh=args['thresh'])
	print('after nms', len(boxes))

	results = []
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	for (start_x, start_y, end_x, end_y) in boxes:
		start_x = int(start_x * r_w)
		start_y = int(start_y * r_h)
		end_x = int(end_x * r_w)
		end_y = int(end_y * r_h)

		d_x = int((end_x - start_x) * args["padding"])
		d_y = int((end_y - start_y) * args["padding"])

		start_x = max(0, start_x - d_x)
		start_y = max(0, start_y - d_y)
		end_x = min(W, end_x + d_x)
		end_y = min(H, end_y + d_y)

		roi_white = image[start_y:end_y, start_x:end_x]
		roi_white = apply_filters(roi_white)
		roi_black = cv2.threshold(roi_white, 127, 255, cv2.THRESH_BINARY_INV)[1]
		# plt.imshow(roi_white, 'gray')
		# plt.show()
		text = max([tess(roi_white), tess(roi_black)], key=lambda x: x[1])[0]
		if text:
			print(text)
			# plt.imshow(roi_white, 'gray')
			# plt.show()
			results.append(((start_x, start_y, end_x, end_y), text))

	# plt.imshow(image, 'gray')
	# plt.show()

	return results


def process(image):
	orig = image.copy()
	if process.count == 0:
		process.results = recognize(image)

	for (start_x, start_y, end_x, end_y), text in process.results:
		cv2.putText(orig, text, (start_x, start_y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
		cv2.rectangle(orig, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
	process.count = (process.count + 1) % 15
	return orig


process.count = 0
process.results = []


def __main__():
	image = cv2.imread(args["image"])
	orig = image.copy()
	results = recognize(image)

	for (start_x, start_y, end_x, end_y), text in results:
		cv2.putText(orig, text, (start_x, start_y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
		cv2.rectangle(orig, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
	plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
	plt.show()


if __name__ == '__main__':
	__main__()
