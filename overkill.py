import argparse
import cv2
import numpy as np
import pytesseract
from imutils.object_detection import non_max_suppression
from matplotlib import pyplot as plt
from pytesseract import Output


def apply_thresholds(orig):
	img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
	gauss_blurs = {
		9: cv2.GaussianBlur(img, (9, 9), 0),
		7: cv2.GaussianBlur(img, (7, 7), 0),
		5: cv2.GaussianBlur(img, (5, 5), 0),
	}
	median_blurs = {
		5: cv2.medianBlur(img, 5),
		3: cv2.medianBlur(img, 3),
	}
	threshes = [
		cv2.threshold(gauss_blurs[9], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
		cv2.threshold(gauss_blurs[7], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
		cv2.threshold(gauss_blurs[5], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
		cv2.threshold(median_blurs[5], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
		cv2.threshold(median_blurs[3], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
		cv2.adaptiveThreshold(gauss_blurs[5], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
		cv2.adaptiveThreshold(median_blurs[3], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
		# cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],
		# cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1],
		# cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1],
		# cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1]
	]

	titles = [
		'gauss9+otsu',
		'gauss7+otsu',
		'gauss5+otsu',
		'median5+otsu',
		'median3+otsu',
		'gauss5+ada_gauss 31 2',
		'gauss3+ada_gauss 31 2',
	]
	return threshes, titles


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


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to input image")
ap.add_argument("-east", "--east", type=str, help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.7, help="minimum confidence")
ap.add_argument("-w", "--width", type=int, default=320, help="nearest multiple of 32 for re-sized width")
ap.add_argument("-e", "--height", type=int, default=320, help="nearest multiple of 32 for re-sized height")
ap.add_argument("-p", "--padding", type=float, default=0.0, help="amount of padding to add to each border of ROI")
ap.add_argument("-l", "--lang", type=str, help="Language", default='eng')
args = vars(ap.parse_args())


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


def __main__():
	image = cv2.imread(args["image"])
	orig = image.copy()
	(origH, origW) = image.shape[:2]

	(newW, newH) = (32 * (origW // 32), 32 * (origH // 32))

	r_w = origW / float(newW)
	r_h = origH / float(newH)

	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

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
	boxes = non_max_suppression(np.array(rectangles), probs=confidences)

	# results = []

	for (start_x, start_y, end_x, end_y) in boxes:
		start_x = int(start_x * r_w)
		start_y = int(start_y * r_h)
		end_x = int(end_x * r_w)
		end_y = int(end_y * r_h)

		d_x = int((end_x - start_x) * args["padding"])
		d_y = int((end_y - start_y) * args["padding"])

		start_x = max(0, start_x - d_x)
		start_y = max(0, start_y - d_y)
		end_x = min(origW, end_x + (d_x * 2))
		end_y = min(origH, end_y + (d_y * 2))

		roi = orig[start_y:end_y, start_x:end_x]
		threshold_images, titles = apply_thresholds(roi)
		for i in range(len(threshold_images)):
			img = tess(threshold_images[i])
			# plt.subplot(3,4,i+1)
			plt.imshow(img)
			plt.title(titles[i])
			plt.show()
			# plt.xticks([]), plt.yticks([])
			print('-' * 10)


if __name__ == '__main__':
	__main__()
