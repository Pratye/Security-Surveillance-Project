import cv2
import numpy as np
import time
import math


Toplam_filtered = []

# Load Yolo-V4 ----------------------------------------------------------------------------------
net = cv2.dnn.readNetFromDarknet("Resources/yolov4-custom-detector.cfg", "Resources/yolov4-custom-detector_best.weights")

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
print(cv2.cuda.getCudaEnabledDeviceCount())

classes = []
with open("Resources/obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Loading video ------------------------------------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # set video width
cap.set(4, 720)  # set video height

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
starting_time = time.time()
frame_id = 0
toplam = 0
say = 0

# Start --------------------------------------------------------------------------------------
while True:
    # Get frame
        _, frame = cap.read()

        frame_id += 1
        if frame_id % 1 != 0:
            continue

        height, width = 720, 1280
        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00261, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        outs = net.forward(output_layers)

        result = []
        boxes = []
        liste = []
        for out in outs:
            for detection in out:
               # print(detection)
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.4:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    cv2.rectangle(frame, (x,y), (x+w, y+h),(0,0,255),1)
                    cv2.imshow('frame', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        # After the loop release the cap object



cap.release()
        # Destroy all the windows
cv2.destroyAllWindows()