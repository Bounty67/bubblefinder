import numpy as np
import cv2
import json
import os
import shutil
import easyocr
import re

def find_circles_in_jpeg(
    file,
    minDist=10,
    param1=10,
    param2=20,
    minRadius=10,
    maxRadius=20
):
    filename = os.path.basename(file)
    base_name = os.path.splitext(filename)[0]
    output_dir = 'JPEGprocessed/' + base_name
    output_dir_temp=os.path.join(output_dir,'temp')
    json_path = os.path.join(output_dir, base_name + ".json")

    # Removing output folder if existing
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_temp, exist_ok=True)

    # Loading source image and detecting balloons
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, minDist,
        param1=param1, param2=param2,
        minRadius=minRadius, maxRadius=maxRadius
    )

    # Initializing OCR
    reader = easyocr.Reader(['fr', 'en', 'de'])

    # Generating output JSON and thumbnails
    filtered_data = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for idx, (x, y, r) in enumerate(circles[0, :], start=1):
            x, y, r = int(x), int(y), int(r)
            # adding detectd circle on source image for preview
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            # thumbnail creation
            roi = img[max(0, y-r):y+r, max(0, x-r):x+r]
            output_path = os.path.join(output_dir_temp, f"{base_name}_{idx}.png")
            cv2.imwrite(output_path, roi)
            # thumbnail pre-processing for OCR
            roi_resized = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            gray_roi = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray_roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            # OCR
            result = reader.readtext(thresh, detail=0)
            text = result[0] if result else ""
            # filtering output for json (here adding specific filter)
            if text == "" or re.fullmatch(r"\d+", text.strip()):
                filtered_data.append({
                    "id": idx,
                    "element": [x, y, r],
                    "text": text
                })
            print(f"Cercle {idx} : '{text}' (retenu : {text == '' or re.fullmatch(r'\\d+', text.strip()) is not None})")

    # creating image from pre-processing
    cv2.imwrite(os.path.join(output_dir_temp, 'preprocessed_' + filename), img)

    # creating image from post-processing (with filters applied)
    img_post = cv2.imread(file)
    for data in filtered_data:
        x, y, r = data["element"]
        cv2.circle(img_post, (x, y), r, (0, 0, 255), 2)
    postprocessed_path = os.path.join(output_dir, f'postprocessed_{filename}')
    cv2.imwrite(postprocessed_path, img_post)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    return filtered_data  #function output when called externally

# arguments processing when called from command line
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Détecte les cercles et extrait les chiffres d'une image.")
    parser.add_argument("file", help="Chemin de l'image à traiter")
    parser.add_argument("--minDist", type=int, default=10, help="Distance minimale entre les centres des cercles")
    parser.add_argument("--param1", type=int, default=10, help="Paramètre 1 pour HoughCircles")
    parser.add_argument("--param2", type=int, default=20, help="Paramètre 2 pour HoughCircles")
    parser.add_argument("--minRadius", type=int, default=10, help="Rayon minimum des cercles")
    parser.add_argument("--maxRadius", type=int, default=20, help="Rayon maximum des cercles")
    args = parser.parse_args()
    find_circles_in_jpeg(
        args.file,
        minDist=args.minDist,
        param1=args.param1,
        param2=args.param2,
        minRadius=args.minRadius,
        maxRadius=args.maxRadius
    )
