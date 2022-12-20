import glob
import cv2 as cv
import numpy as np
#import crope
#import matplotlib.pyplot as plt
import time
# from io import BytesIO
# from google.cloud import storage

threshold = 0.5
sup_threshold = 0.4
yolo_size = 320
font = cv.FONT_HERSHEY_COMPLEX_SMALL
color = (0, 255, 0)
detected_object_list = []

# Function which find the object Bounding Box with class confidence
# find the predicted objects
def finding_objects(yolo_outputs):
    class_ids = []
    confidence_values = []
    bounding_boxes_locations = []
    # for each output layer we keep the bb with high confidence value
    for output in yolo_outputs:
        for prediction in output:
            # as we know bb contain [x,y,w,h,confi] + 80 classes
            object_score = prediction[5:]
            class_id = np.argmax(object_score)
            confidence = object_score[class_id]
            # getting rid of all the bb whose confidence is less THRESHOLD
            if confidence > threshold:
                # transforming the bb values
                w, h = int(prediction[2] * yolo_size), int(prediction[3] * yolo_size)
                x, y = int(prediction[0] * yolo_size - w / 2), int(prediction[1] * yolo_size - h / 2)
                # keeping all BB of high confidence
                bounding_boxes_locations.append([x, y, w, h, ])
                class_ids.append(class_id)
                confidence_values.append(float(confidence))
    # non maximum suppression used to get rid of multiple detected object BB
    boxes_to_keep = cv.dnn.NMSBoxes(bounding_boxes_locations, confidence_values, threshold, sup_threshold)
    return boxes_to_keep, bounding_boxes_locations, class_ids, confidence_values


def draw_rectangle(image, bounding_box_ids, all_bb_loca, class_ids, confidence, class_labels, width_ratio,
                   height_ratio):
    class_name = []
    for box in range(len(all_bb_loca)):

        if box in bounding_box_ids:
            x, y, w, h = all_bb_loca[box]
            x = int(x * width_ratio)
            y = int(y * height_ratio)
            w = int(w * width_ratio)
            h = int(h * height_ratio)

            class_name.append(class_labels[class_ids[box]])
            class_label = class_labels[class_ids[box]] + " " + str(round(float(confidence[box] * 100), 2)) + '%'
            cv.rectangle(image, (x, y), (x + w, y + h), color, 3)
            cv.putText(image, class_label, (x, y - 10), font, 0.9, (0, 0, 255), 1)
    Class_Names = List_of_detectObject(class_name)
    return Class_Names


# function Reading class name
def Reading_className():
    classes = []
    with open("classes.txt", "r") as file:
        for line in file:
            for word in line.split():
                classes.append(word)
    return classes


def List_of_detectObject(classes):
    for i in range(len(classes)):
        if classes[i] not in detected_object_list:
            detected_object_list.append(classes[i])

    return detected_object_list


if __name__ == "__main__":
    # loading  Yolov3.weights and yolov3.cfg and making a network
    neural_network = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
    neural_network.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    neural_network.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    count = 0
    Classes = Reading_className()
    cam = cv.VideoCapture(0)
    while True:
        ret,frame = cam.read()
        count = count+1
        starting_time = time.time()
        #image = cv.imread("cam.jpeg")
        #print(type(frame))
        # storing the original image size
        [org_image_height, org_image_width, channels] = frame.shape
        # resizing the high dimensional images to normal (width 720 ,height 1080)
        width2 = 700
        height2 = 400
        # Setting the points for cropped image
        # our network take blob of size 320x320 so we use blob function
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (yolo_size, yolo_size), True, crop=False)
        
        neural_network.setInput(blob)
        # give the list of all layers
        layer_names = neural_network.getLayerNames()
        # give the index of output layers
        
        #output_layers = [layer_names[i- 1] for i in neural_network.getUnconnectedOutLayers()]
        output_layers = neural_network.getUnconnectedOutLayersNames()
        #print("layers name: ",output_layers)
        # forward propagation return output predictions containing bounding box,w,h,x,y,confi,and class id
        outputs = neural_network.forward(output_layers)
        # print(outputs[0].shape)
        predicted_objects, BBox_location, class_label_ids, confi_values = finding_objects(outputs)

        Totla_Classes_Predicted = draw_rectangle(frame, predicted_objects, BBox_location, class_label_ids,
                                                 confi_values, Classes,
                                                 org_image_width / yolo_size, org_image_height / yolo_size)

        elapsed_time = time.time() - starting_time

        fps = frame/elapsed_time
        cv.putText(frame,"FPS: "+str(np.round(fps,2)),(50,50),font,1,(0, 0, 0),1)
        # resized_image = cv.resize(frame, (width2, height2))
        cv.imshow("yolo", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        


    # After the loop release the cap object
    cam.release()
    # Destroy all the windows
    cv.destroyAllWindows()
    print(Totla_Classes_Predicted)






