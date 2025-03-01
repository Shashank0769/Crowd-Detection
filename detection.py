import cv2
import pandas as pd
import numpy as np


net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()

output_layer_indices = net.getUnconnectedOutLayers()

if isinstance(output_layer_indices, np.ndarray):
    output_layer_indices = output_layer_indices.flatten().tolist()

output_layers = [layer_names[i - 1] for i in output_layer_indices]
def detect_crowds(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    crowd_events = []
    person_positions = []
    consecutive_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        height, width, _ = frame.shape
        
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        persons = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    persons.append((center_x, center_y, w, h))

        if len(persons) >= 3:
            close_groups = []
            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    if np.linalg.norm(np.array(persons[i][:2]) - np.array(persons[j][:2])) < 100:
                        close_groups.append(persons[i])
                        close_groups.append(persons[j])
            close_groups = list(set(close_groups))

            if len(close_groups) >= 3:
                consecutive_frames += 1
                if consecutive_frames == 10:
                    crowd_events.append((frame_count, len(close_groups)))
            else:
                consecutive_frames = 0
        else:
            consecutive_frames = 0

    cap.release()
    return crowd_events
def save_results(crowd_events, output_file):
    df = pd.DataFrame(crowd_events, columns=["Frame Number", "Person Count in Crowd"])
    df.to_csv(output_file, index=False)


video_path = r"C:\Users\shash\crowd_detection\sample.mp4"
output_file = "crowd_detection_results.csv"

crowd_events = detect_crowds(video_path)
save_results(crowd_events, output_file)