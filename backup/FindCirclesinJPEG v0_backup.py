import numpy as np
import cv2
import json
import os
import shutil
import easyocr

reader = easyocr.Reader(['fr', 'en', 'de'])

## json Export function
def export_array_to_json(array, filename):
    data = []
    for idx, item in enumerate(array, start=1):
        obj = {
            "id": idx,
            "element": [int(x) for x in item],
            "text": ""
        }
        data.append(obj)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# variables
file = 'JPEGoutput/vueEclatee.jpeg'
filename = os.path.basename(file)
base_name = os.path.splitext(os.path.basename(file))[0]
output_dir = 'JPEGprocessed/' + base_name + "_circles"
json_path = os.path.join(output_dir, base_name + ".json")

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(output_dir, exist_ok=True)

#prog
img = cv2.imread(file)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)

minDist = 10
param1 = 10 #500
param2 = 20 #50 #200 #smaller value-> more false circles (12 working fine)
minRadius = 10
maxRadius = 20 #100 #10

circles = cv2.HoughCircles(gray,
                           cv2.HOUGH_GRADIENT,
                            1,
                            minDist,
                            param1=param1,
                            param2=param2,
                            minRadius=minRadius,
                            maxRadius=maxRadius)

if circles is not None:
    circles = np.uint16(np.around(circles))
    id=0
    export_array_to_json(circles[0], json_path)
    for i in circles[0,:]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)


# Saving resulting picture
cv2.imwrite(output_dir + '/processed_' + filename, img)

for idx, (x, y, r) in enumerate(circles[0, :], start=1):
    x, y, r = int(x), int(y), int(r)
    # creating square zone around image
    roi = img[max(0, y-r):y+r, max(0, x-r):x+r]
    # creating thumbnails
    output_path = os.path.join(output_dir, f"{base_name}_{idx}.png")
    cv2.imwrite(output_path, roi)
    #trying to apply OCR on thumbnail
    roi = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    result = reader.readtext(thresh, detail=0)
    print(f"Cercle {idx} : {result}")


# Show result for testing:
#cv2.imshow('img', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

