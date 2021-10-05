import cv2
import numpy as np
import os


# Load Yolo-V4 ----------------------------------------------------------------------------------
net = cv2.dnn.readNetFromDarknet("Resources/yolov4-custom-detector.cfg", "Resources/yolov4-custom-detector_best.weights")

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
print(cv2.cuda.getCudaEnabledDeviceCount())

classes = []
with open("Resources/obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
# generate different colors for different classes
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Loading video ------------------------------------------------------------------------
cap = cv2.VideoCapture('Resources/input_video.MOV')
cap.set(3, 1280)  # set video width
cap.set(4, 720)  # set video height

#output_video = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1280,720))
output_video = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (1280,720))
# Start --------------------------------------------------------------------------------------
while True:
    try:
        # Get frame
        _, frame = cap.read()
        #height, width = 720, 1280

        #frame = cv2.imread('Resources/Person.jpg')
        height = frame.shape[0]
        width = frame.shape[1]

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00261, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        bboxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    bboxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(bboxes, confidences, 0.5, 0.4)

        for i in indices:
            i = i[0]
            bbox = bboxes[i]
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]

            label = str(classes[class_ids[i]])

            #color = colors[class_ids[i]]

            x = round(x)
            y = round(y)
            x_plus_w = round(x+w)
            y_plus_h = round(y+h)


            if label=='obama':
                status = 'Status: Access Approved!'
                label  = 'Obama'
                color = (0,255,0)
                status_color = (0,255,0)
            elif label=='pistol' or label=='knife':
                status = 'Status: Access Denied! Calling Police...'
                color = (0, 0, 255)
                status_color = (0, 0, 255)
            elif label=='person':
                status = 'Status: Access Denied!'
                label = 'Unidentified Person'
                color = (0, 0, 255)
                status_color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x_plus_w, y_plus_h), color, 3)
            cv2.putText(frame, label, (x - 10, y - 10), font, 2, color, 2)
            cv2.putText(frame, status, (round(0.05*width), round(0.9*height)), font, 2, status_color, 2)
           # cv2.imshow('result', frame)
            output_video.write(frame)
            '''if cv2.waitKey(1) & 0xFF == ord('q'):
                break'''
            '''if cv2.waitKey(0):
                break'''
    except:
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()

def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = avi_file_path, output = output_name))
    return True
#convert_avi_to_mp4('Resources/out.avi', 'Resources/output_video.mp4')