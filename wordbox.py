import pytesseract
from pytesseract import Output
import cv2
import argparse
from googletrans import Translator
from matplotlib import pyplot as plt
import numpy as np

translator = Translator()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to input image")
ap.add_argument('-l', '--lang', type=str, help='Language', default='eng')
args = vars(ap.parse_args())

orig = cv2.imread(args["image"])


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
		cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],
		cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1],
		cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1],
		cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1]
	]
	return threshes

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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


th_imgs = thressher(orig)
for i in range(len(th_imgs)):
	img = tess(th_imgs[i])
	# plt.subplot(3,4,i+1)
	plt.imshow(img)
	plt.show()
	# plt.title(titles[i])
	# plt.xticks([]), plt.yticks([])
	print('-'*10)




# plt.imshow()


# cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)
# cv2.imshow('image', cv2.resize(img, (720, 480)))
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
# cv2.waitKey(0)





