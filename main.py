import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from PIL import Image
from imutils import contours
from keras.engine.saving import model_from_json
from matplotlib import gridspec
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import LabelEncoder
from os.path import splitext, basename
from local_utils import detect_lp

# Load Yolo
net = cv2.dnn.readNet("yolov3_custom_last.weights", "yolov3_custom.cfg")
classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = cv2.imread("TestData/TestCar05.jpg")

height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (608, 608), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Show information on the screen
class_ids = []
confidences = []
boxes = []

tmp_class_ids = []
tmp_confidences = []
tmp_boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            # Object detected
            print(confidence)
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2) - 5
            y = int(center_y - h / 2) - 5

            tmp_boxes.append([x, y, w, h])
            tmp_confidences.append(float(confidence))
            tmp_class_ids.append(class_id)

maxindex = np.argmax(tmp_confidences)
boxes.append(tmp_boxes[maxindex])
confidences.append(tmp_confidences[maxindex])
class_ids.append(tmp_class_ids[maxindex])

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
isYes = 0
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        crop_img = img[y:y + h + 10, x:x + w + 15]
        crop_img2 = cv2.resize(crop_img, (400, 200), fx=1, fy=1)
        cv2.imshow("LicensePlate", crop_img2)
        cv2.imwrite('tmp/LicensePlate.jpg', crop_img2, [cv2.IMWRITE_JPEG_QUALITY, 90])
        # label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        # cv2.putText(img, label, (x, y + 30), font, 1, color, 3)
        if class_ids[i] == 0:
            isYes = 0
        else:
            isYes = 1

def secondCrop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('Second crop plate1', thresh)
    areas = [cv2.contourArea(c) for c in contours]
    if (len(areas) != 0):
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        secondCrop = img[y:y + h, x:x + w]
    else:
        secondCrop = img
    return secondCrop

img = cv2.resize(img, (600, 400))
cv2.imshow("Image", img)
cv2.imwrite('tmp/firstImg.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])

secondCropImg = secondCrop(crop_img2)

secondCropImg = cv2.resize(secondCropImg, (400, 200))
cv2.imshow('Second crop plate', secondCropImg)
cv2.waitKey(0)
cv2.imwrite('tmp/secondCropImg.jpg', secondCropImg, [cv2.IMWRITE_JPEG_QUALITY, 90])
# def load_model(path):
#     try:
#         path = splitext(path)[0]
#         with open('%s.json' % path, 'r') as json_file:
#             model_json = json_file.read()
#         model = model_from_json(model_json, custom_objects={})
#         model.load_weights('%s.h5' % path)
#         print("Loading model successfully...")
#         return model
#     except Exception as e:
#         print(e)
#
# wpod_net_path = "wpod-net/wpod-net.json"
# wpod_net = load_model(wpod_net_path)
# def preprocess_image(image_path,resize=False):
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = img / 255
#     if resize:
#         img = cv2.resize(img, (224,224))
#     return img
#
# def get_plate(image_path, Dmax=608, Dmin = 608):
#     vehicle = preprocess_image(image_path)
#     ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
#     side = int(ratio * Dmin)
#     bound_dim = min(side, Dmax)
#     _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
#     return vehicle, LpImg, cor
#
# test_image_path = "tmp/firstImg.jpg"
# vehicle, LpImg,cor = get_plate(test_image_path)
#
# fig = plt.figure(figsize=(12,6))
# grid = gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
# fig.add_subplot(grid[0])
# plt.axis(False)
# plt.imshow(vehicle)
# grid = gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
# fig.add_subplot(grid[1])
# plt.axis(False)
# plt.imshow(LpImg[0])

# turn grey
gray = cv2.cvtColor(secondCropImg, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 1)

# turn binary
binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
edged = cv2.Canny(thre_mor, 1, 200)
ret, threshed = cv2.threshold(edged, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
contours, _ = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cv2.imwrite('tmp/cannyImg.jpg', edged, [cv2.IMWRITE_JPEG_QUALITY, 90])
# the top position of car plate
car_plate = []

contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# find the car plate location(the maximum rectangle(width>70%) on image , and save to car_plate
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if w / edged.shape[1] >= 0.8:  # 僅width>0.8的輪廓
        car_plate = approx.tolist()
        cv2.drawContours(binary, np.array(car_plate), -1, (0, 255, 0), 3)

if not car_plate:  # if car_plate is empty
    print("No car plate found")
    os._exit(0)

# for x in range(0, 4):
#      cv2.circle(binary, (car_plate[x][0][0], car_plate[x][0][1]), 5, (0, 0, 255), cv2.FILLED)
#      cv2.putText(binary, '%d' % x, (car_plate[x][0][0], car_plate[x][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

# plate_width=1200
# plate_height=600

plate_width = car_plate[3][0][0] - car_plate[0][0][0]
plate_height = car_plate[2][0][1] - car_plate[3][0][1]

pts1 = np.float32([[car_plate[0][0][0], car_plate[0][0][1] - 10], [car_plate[1][0][0], car_plate[1][0][1] + 10],
                   [car_plate[2][0][0], car_plate[2][0][1] + 10], [car_plate[3][0][0], car_plate[3][0][1] + 10]])

pts2 = np.float32([[0, 0], [0, plate_height], [plate_width, plate_height], [plate_width, 0]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutput = cv2.warpPerspective(binary, matrix, (plate_width, plate_height))


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


# print(imgOutput[0],imgOutput[1] )
##img_height=imgOutput[1]
# plate_ratio = img_height / img_width
# if 0.05>plate_ratio<=0: #if width>height , rotate angle=90
#   imgOutput = rotate_bound(imgOutput, 90)
#   imgOutput=cv2.resize(imgOutput, dsize=(img_width , img_height))


cv2.imshow("Perspective transformation", imgOutput)
os.system("rm -rf tmp")
os.mkdir("tmp", 0o777);

# save the Perspective plate image to /tmp
# platenum=len(boxes)

cv2.imwrite('tmp/detectPlate.jpg', imgOutput, [cv2.IMWRITE_JPEG_QUALITY, 90])


# if isYes ==0:
#     text = pytesseract.image_to_string(imgOutput, lang='eng', config='-c tessedit_char_whitelist=01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7')
#     print(text, "hi, single")
# else:
#     text = pytesseract.image_to_string(imgOutput, lang='eng', config='-c tessedit_char_whitelist=01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6')
#     print(text, "hi")

# 創建sort_contours（）函數以從左到右抓取每個數字的輪廓
def sort_contours(cnts, reverse=False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]  # 獲取最小外接矩形,[x,y,w,h]
    (cnts, boundingBoxes) = zip(
        *sorted(zip(cnts, boundingBoxes),  # 合二為一zip([digit1,boundingBoxes1],[digit2,boundingBoxes2])
                key=lambda b: b[1][i], reverse=reverse))  # 按_順序,再組成list
    return cnts


plateImg = cv2.imread('tmp/detectPlate.jpg')  # read image and become threshold to detect cnts
gray = cv2.cvtColor(plateImg, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)

ret, threshed = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cnts, _ = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 用二值圖檢測輪廓

# create a copy version "test_digit" of plateImg to draw bounding box
test_digit = plateImg.copy()

# Initialize a list which will be used to append charater image
crop_characters = []

for c in sort_contours(cnts):
    x, y, w, h = cv2.boundingRect(c)
    ratio = h / w

    if 1 <= ratio <= 9:  # 僅選擇定義比例的輪廓
        if isYes == 0:
            if h / plateImg.shape[0] >= 0.4 and h / plateImg.shape[0] < 0.92:  # 選擇高度大於plate的40％-92%的輪廓
                # 在數字周圍繪製邊界框
                cv2.rectangle(test_digit, (x, y), (x + w, y + h), (0, 255, 0), 1)

                # Separate number and gibe prediction
                curr_num = blur[y:y + h, x:x + w]
                curr_num = cv2.resize(curr_num, dsize=(w + 10, h + 10))
                _, curr_num = cv2.threshold(curr_num, 200, 255, cv2.THRESH_BINARY)
                crop_characters.append(curr_num)
        else:
            if h / plateImg.shape[0] >= 0.3 and h / plateImg.shape[0] < 0.82:  # 選擇高度大於plate的40％-92%的輪廓
                # 在數字周圍繪製邊界框
                cv2.rectangle(test_digit, (x, y), (x + w, y + h), (0, 255, 0), 1)

                # Separate number and gibe prediction
                curr_num = blur[y:y + h, x:x + w]
                curr_num = cv2.resize(curr_num, dsize=(w + 10, h + 10))
                _, curr_num = cv2.threshold(curr_num, 200, 255, cv2.THRESH_BINARY)
                crop_characters.append(curr_num)

print("Detect {} letters...".format(len(crop_characters)))
cv2.imshow("detectDigit", test_digit)
cv2.imwrite('tmp/detectDigit.jpg', test_digit, [cv2.IMWRITE_JPEG_QUALITY, 100])

# Load model architecture, weight and labels
json_file = open('MobileNets/MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("MobileNets/License_character_recognition_weight.h5")
# print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('MobileNets/license_character_classes.npy')


# print("[INFO] Labels loaded successfully...")

# 預處理輸入圖像和帶模型的Pedict
def predict_from_model(image, model, labels):
    image = cv2.resize(image, (80, 80))
    image = np.stack((image,) * 3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis, :]))])
    return prediction


fig = plt.figure(figsize=(15, 3))
cols = len(crop_characters)
grid = gridspec.GridSpec(ncols=cols, nrows=1, figure=fig)

final_string = ''
for i, character in enumerate(crop_characters):  # 索引序列
    fig.add_subplot(grid[i])
    title = np.array2string(predict_from_model(character, model, labels))
    plt.title('{}'.format(title.strip("'[]"), fontsize=20))
    final_string += title.strip("'[]")
    plt.axis(False)
    plt.imshow(character, cmap='gray')

print(final_string)
plt.savefig('final_result.png', dpi=300)

size = cv2.getTextSize(final_string, font, 2, 2)
text_width = size[0][0]
text_height = size[0][1]

cv2.rectangle(img, (boxes[0][0], boxes[0][1] - text_height), (boxes[0][0] + text_width, boxes[0][1]), (255, 255, 255),
              -1)
cv2.putText(img, final_string, (boxes[0][0], boxes[0][1]), font, 2, (0, 0, 0), 2)
cv2.imshow("result", img)
cv2.imwrite('tmp/result.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
cv2.waitKey(0)
cv2.destroyAllWindows()
