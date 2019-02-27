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

img = cv2.imread(args["image"])
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# cv2.imshow('c', img)
# plt.show()

d = pytesseract.image_to_data(img, output_type=Output.DICT, lang=args['lang'])
n_boxes = len(d['level'])

for i in range(n_boxes):
	if int(d['conf'][i]) < 75 or not d['text'][i]:
		continue
	(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv2.putText(img, translator.translate(d['text'][i]).text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
	print(str(i) + ':' + d['text'][i])


cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)
# cv2.imshow('image', cv2.resize(img, (720, 480)))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
plt.show()
# cv2.waitKey(0)





