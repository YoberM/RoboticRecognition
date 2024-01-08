import cv2
import numpy as np
import argparse

data = {"nothing"}
dataConfiable = {"nothing"}

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def get_labels(image ):

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open("yolov3.txt", 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    blob = cv2.dnn.blobFromImage(image, scale, (608,608), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    detected_classes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    font = cv2.FONT_ITALIC=16
    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]
        
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        cv2.rectangle(image, (round(x), round(y)), (round(x+w), round(y+h)),COLORS[class_ids[i]],1)
        cv2.putText(image, (classes[class_ids[i]] + str(round(confidences[i], 3))), (round(x), round(y) + 30), font, 1, COLORS[class_ids[i]], 2)
        
        data.add(classes[class_ids[i]])
        if (confidences[i] > 0.9):
            detected_classes += [{'class' : classes[class_ids[i]], 'confidence': confidences[i] }]
            dataConfiable.add(classes[class_ids[i]])

    return detected_classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Procesar archivo de video.')
    parser.add_argument('archivo', help='Ruta del archivo de video')
    
    args = parser.parse_args()

    video_dir = "Videos/"
    video_name = args.archivo
    video_nameOut = video_name
    video = cv2.VideoCapture(video_dir + video_name)
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    output_video = cv2.VideoWriter(video_nameOut, fourcc, fps, (width, height))

    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    ret , frame = video.read()
    print(frames,fps)
    count = 0
    efps = 1

    video_labels = {}
    while (ret):
        count += 1
        ## This gets labels each 24/30/60 fps
        # if count % int(fps/efps) == 0 or count % 15 == 0:
        if count % int(fps/efps) == 0:
            print(count)
            labels = get_labels(frame)
            for label in labels:
                if label['class'] in video_labels:
                    if video_labels[label['class']]['confidence'] < label['confidence']:
                        video_labels[label['class']]['image'] = frame
                        video_labels[label['class']]['confidence'] = label['confidence']
                else:
                    video_labels[label['class']] = {
                        'image': frame,
                        'confidence' : label['confidence']
                    }
            output_video.write(frame)
        ret , frame = video.read()
    output_video.release()
    video.release()
    print(data)
    print(dataConfiable)