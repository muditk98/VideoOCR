import pytesseract as pt
import argparse
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np

fontpath = "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"
ttffont = ImageFont.truetype(fontpath, 64)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to input image")
ap.add_argument('-l', '--lang', type=str, help='Language', default='eng')
args = vars(ap.parse_args())

img = cv2.imread(args["image"])
h, w, _ = img.shape  # assumes color image

print(pt.image_to_data(img, lang=args['lang']))

'''
# run tesseract, returning the bounding boxes
boxes = pt.image_to_boxes(img, lang='deu')  # also include any config options you use

# draw the bounding boxes on the image
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
    # cv2.putText(img, b[0], (int(b[1]), h - int(b[2]) - 60), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 3)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((int(b[1]), h - int(b[2]) - 60), b[0], font=ttffont, fill=(0, 0, 255, 0))
    img = np.array(img_pil)
    print(b[0])

# show annotated image and wait for keypress
cv2.imshow("ww", cv2.resize(img, (720, 480)))
cv2.waitKey(0)
'''

