import glob
import cv2 as cv
import numpy as np
#import crope

#from pydrive.auth import  GoogleAuth
#from pydrive.drive import GoogleDrive



_threshold = 0.5
_NMS_threshold = 0.4
_yolo_size = 410
font = cv.FONT_HERSHEY_COMPLEX_SMALL
color = (0, 255, 0)
detected_object_list = []



# Function which find the object Bounding Box with class confidence
# find the predicted objects
def finding_objects(yolo_outputs):
    print("yolo outputs: ",yolo_outputs[0].shape,yolo_outputs[1].shape,yolo_outputs[2].shape)
    class_ids = []
    confidence_values = []
    bounding_boxes_locations = []
    # for each output layer we keep the bb with high confidence value
    for output in yolo_outputs:
        for prediction in output:
            # as we know bb contain [x,y,w,h,confi] + 80 classes
            # print(prediction.shape)

            object_score = prediction[5:]
            # print("object_score",object_score)
            class_id = np.argmax(object_score)
            # print("class_id",class_id)
            confidence = object_score[class_id]
            # print("confidence",confidence)

            # getting rid of all the bb whose confidence is less THRESHOLD
            if confidence > _threshold:
                # transforming the bb values
                w, h = int(prediction[2] * _yolo_size), int(prediction[3] * _yolo_size)
                x, y = int(prediction[0] * _yolo_size - w / 2), int(prediction[1] * _yolo_size - h / 2)
                # keeping all BB of high confidence
                bounding_boxes_locations.append([x, y, w, h, ])
                class_ids.append(class_id)
                confidence_values.append(float(confidence))

    # non maximum suppression used to get rid of multiple detected object BB
    boxes_to_keep = cv.dnn.NMSBoxes(bounding_boxes_locations, confidence_values, _threshold, _NMS_threshold)
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


# function reading images
def Reading_Images():
    path = glob.glob("tv.jpg")
    images = [cv.imread(img) for img in path]
    return images

def List_of_detectObject(classes):
    for i in range(len(classes)):
        if classes[i] not in detected_object_list:
            detected_object_list.append(classes[i])

    return detected_object_list

if __name__ == "__main__":
    # loading  Yolov3.weights and yolov3.cfg and making a network
    neural_network = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
    # give the list of all layers
    layer_names = neural_network.getLayerNames()
    # print(layer_names)
    # give the index of output layers
    output_layers = [layer_names[i[0] - 1] for i in neural_network.getUnconnectedOutLayers()]
    # print(output_layers)
    Classes = Reading_className()
    Images = Reading_Images()

    for image in range(len(Images)):
       
        # resizing the high dimensional images to normal (width 720 ,height 1080)
        IMAGE = crope.Resizing_image(Images[image])
        # storing the original image size
        [org_image_height, org_image_width, channels] = IMAGE.shape
        # our network take blob of size 320x320 so we use blob function
        blob = cv.dnn.blobFromImage(IMAGE, 1 / 255, (_yolo_size, _yolo_size), True, crop=False)
        neural_network.setInput(blob)
        # forward propagation return output predictions containing bounding box,w,h,x,y,confi,and class id
        outputs = neural_network.forward(output_layers)
        print("lenght: ",len(outputs))
        predicted_objects, BBox_location, class_label_ids, confi_values = finding_objects(outputs)

        Totla_Classes_Predicted = draw_rectangle(IMAGE, predicted_objects, BBox_location, class_label_ids, confi_values, Classes,
                                                 org_image_width / _yolo_size, org_image_height / _yolo_size)


        # cv.imwrite("pred.jpeg", IMAGE)
        cv.imshow("yolo", IMAGE)
        cv.waitKey()
        cv.destroyAllWindows()

print(Totla_Classes_Predicted)
textfile = open("Detected Objects.txt", "w")
for element in Totla_Classes_Predicted:
    textfile.write(element + "\n")
textfile.close()

#gauth = GoogleAuth()

#gauth.LocalWebserverAuth()

#drive = GoogleDrive(gauth)

#file1 = drive.CreateFile()
#file1.SetContentFile("Detected Objects.txt")
#file1.Upload()
