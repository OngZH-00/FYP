from flask import Flask, render_template, request, jsonify
from matplotlib import pyplot as plt
import cv2
import numpy as np
import imutils
import easyocr
import base64
import datetime


app = Flask(__name__)


@app.route('/')
@app.route('/detection')
def detection():
    return render_template('detection.html')
#
#


# @app.route('/comparation')
# def comparation():
#     return render_template('comparation.html')
#
#


# @app.route('/hello', methods=['POST'])
# def hello():
#     first_name = request.form['first_name']
#     last_name = request.form['last_name']
#     image = request.form['image']
#     return 'Hello %s %s have fun learning python <br/> <a href="/">Back Home</a> <img src="%s" alt="Image Preview" class="image-preview__image">' % (first_name, last_name, image)
# #
# #


@app.route('/detect-car-plate', methods=['POST'])
def detect_car_plate():
    try:
        image_base64 = request.form['imageBase64']
        image_raw = image_base64.split(',')[1]
        result = get_car_plate(image_raw)

    except Exception as exception:
        print('{0} Error detect_car_plate - {1}'.format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            str(exception)
        ))
        result = exception

    finally:
        return render_template('detection.html', result=result)
#
#


@app.route('/compare-car-plate', methods=['POST'])
def compare_car_plate():
    try:
        car_plate = request.form['carPlate']
        image_base64 = request.form['imageBase64']
        image_raw = image_base64.split(',')[1]

        result = car_plate == get_car_plate(image_raw)

    except Exception as exception:
        print('{0} Error compare_car_plate - {1}'.format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            str(exception)
        ))
        result = exception

    finally:
        return jsonify({'result': result})
#
#


def get_car_plate(image_raw):
    try:

        with open("image_to_process.jpg", "wb") as f:
            f.write(base64.b64decode(image_raw))
        img = cv2.imread('image_to_process.jpg')
        # if img.size == 0:
        #     print('{0} Error get_car_plate cannot read image'.format(
        #         datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        #     ))
        #     result = 'Invalid Image Provided'
        # else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
        edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
        plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
        keypoints = cv2.findContours(
            edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)
        plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        text = result[0][-2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        res = cv2.putText(img, text=text, org=(
            approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        res = cv2.rectangle(img, tuple(approx[0][0]), tuple(
            approx[2][0]), (0, 255, 0), 3)
        plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        print('{0} Complete get_car_plate, Car Plate Number Is {1}'.format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            text
        ))
        result = text

    except Exception as exception:
        print('{0} Error get_car_plate - {1}'.format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            str(exception)
        ))
        result = exception

    finally:
        return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
