
import cv2
import numpy as np
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
from imutils import contours
import os, sys

# Load Yolo
from keras.engine.saving import model_from_json
from matplotlib import gridspec
from sklearn.preprocessing import LabelEncoder

net = cv2.dnn.readNet("yolov3_custom_best.weights", "yolov3_custom.cfg")
classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = cv2.imread("test/t12.png")

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
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            tmp_boxes.append([x, y, w, h])
            tmp_confidences.append(float(confidence))
            tmp_class_ids.append(class_id)

maxindex = np.argmax(tmp_confidences)
boxes.append(tmp_boxes[maxindex])
confidences.append(tmp_confidences[maxindex])
class_ids.append(tmp_class_ids[maxindex])

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        crop_img = img[y:y + h, x:x + w]
        crop_img2 = cv2.resize(crop_img, (400, 200), fx=1.5, fy=1.5)
        cv2.imshow("LicensePlate", crop_img2)

        # label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        # cv2.putText(img, label, (x, y + 30), font, 1, color, 3)
img = cv2.resize(img, (600, 400))
cv2.imshow("Image", img)

# turn grey
gray = cv2.cvtColor(crop_img2, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 1)
# turn binary
binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
ret, threshed = cv2.threshold(binary, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
contours, _ = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


# the top position of car plate
car_plate = []

contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# find the car plate location(the maximum rectangle(width>70%) on image , and save to car_plate
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) == 4  and w / binary.shape[1] >= 0.7:  # 僅width>0.7的輪廓
        car_plate = approx.tolist()
        cv2.drawContours(binary, np.array(car_plate), -1, (0, 255, 0), 3)

if not car_plate: # if car_plate is empty
    print("No car plate found")
    os._exit(0)



for x in range(0, 4):
    cv2.circle(binary, (car_plate[x][0][0], car_plate[x][0][1]), 5, (0, 0, 255), cv2.FILLED)
    cv2.putText(binary, '%d' % x, (car_plate[x][0][0], car_plate[x][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

# plate_width=1200
# plate_height=600

plate_width = car_plate[3][0][0] - car_plate[0][0][0]
plate_height = car_plate[2][0][1] - car_plate[3][0][1]

pts1 = np.float32([[car_plate[0][0][0], car_plate[0][0][1]], [car_plate[1][0][0], car_plate[1][0][1]],
                   [car_plate[2][0][0], car_plate[2][0][1]], [car_plate[3][0][0], car_plate[3][0][1]]])

pts2 = np.float32([[0, 0], [0, plate_height], [plate_width, plate_height], [plate_width, 0]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutput = cv2.warpPerspective(binary, matrix, (plate_width, plate_height))

cv2.imshow("Perspective transformation", imgOutput)
os.system("rm -rf tmp")
os.mkdir("tmp",0o777);

#save the Perspective plate image to /tmp
#platenum=len(boxes)

cv2.imwrite('tmp/detectPlate.jpg', imgOutput, [cv2.IMWRITE_JPEG_QUALITY, 90])


# 創建sort_contours（）函數以從左到右抓取每個數字的輪廓
def sort_contours(cnts, reverse=False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]  # 獲取最小外接矩形,[x,y,w,h]
    (cnts, boundingBoxes) = zip(
        *sorted(zip(cnts, boundingBoxes),  # 合二為一zip([digit1,boundingBoxes1],[digit2,boundingBoxes2])
                key=lambda b: b[1][i], reverse=reverse))  # 按_順序,再組成list
    return cnts

plateImg = cv2.imread('tmp/detectPlate.jpg') #read image and become threshold to detect cnts
gray = cv2.cvtColor(plateImg,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(7,7),0)
ret, threshed = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
cnts, _  = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #用二值圖檢測輪廓

# creat a copy version "test_digit" of plateImg to draw bounding box
test_digit = plateImg.copy()

# Initialize a list which will be used to append charater image
crop_characters = []


for c in sort_contours(cnts):
    x, y, w, h = cv2.boundingRect(c)
    ratio = h / w

    if 1 <= ratio <= 6.5:  # 僅選擇定義比例的輪廓
        if h / plateImg.shape[0] >= 0.4 and h / imgOutput.shape[0] < 0.92:  # 選擇高度大於plate的40％-92%的輪廓
            # 在數字周圍繪製邊界框
            cv2.rectangle(test_digit, (x, y), (x + w, y + h ), (0, 255, 0), 1)

            # Sperate number and gibe prediction
            curr_num = blur[y:y + h , x:x + w ]
            curr_num = cv2.resize(curr_num, dsize=(w+10 , h+10))
            _, curr_num = cv2.threshold(curr_num, 200, 255, cv2.THRESH_BINARY)
            crop_characters.append(curr_num)

print("Detect {} letters...".format(len(crop_characters)))
cv2.imshow("detectDigit", test_digit)

# Load model architecture, weight and labels
json_file = open('MobileNets/MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("MobileNets/License_character_recognition_weight.h5")
#print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('MobileNets/license_character_classes.npy')
#print("[INFO] Labels loaded successfully...")

# 預處理輸入圖像和帶模型的Pedict
def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction


final_string = ''
for i,character in enumerate(crop_characters): #索引序列
    title = np.array2string(predict_from_model(character,model,labels))
    final_string+=title.strip("'[]")

print(final_string)

size = cv2.getTextSize(final_string, font, 2, 2)
text_width = size[0][0]
text_height = size[0][1]

cv2.rectangle(img, (boxes[0][0], boxes[0][1]-text_height), (boxes[0][0]+text_width, boxes[0][1]), (255,255,255), -1)
cv2.putText(img, final_string, (boxes[0][0], boxes[0][1]), font, 2, (0,0,0), 2)
cv2.imshow("result",img)
cv2.waitKey(0)
cv2.destroyAllWindows()