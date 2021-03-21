
# Imports
import cv2
import numpy as np
from model import SSD


BOX_WIDTH_MIN = 120
BOX_HEIGHT_MIN = 80

BOX_WIDTH_MAX = 1200
BOX_HEIGHT_MAX = 800

OFFSET = 8

	
def get_centroid(x, y, w, h):
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    return cx, cy


def main(video_path, line_y_position):

    cap = cv2.VideoCapture(video_path)
    model = SSD("weights/ssd_mobilenet_v1_fpn.pb")

    vehicle_count= 0
    detections = []

    while True:
        
        ret, frame = cap.read()

        image_height, image_width, _ = frame.shape
        if not ret:
            break
        
        cv2.line(frame, (0, line_y_position), (image_width, line_y_position), (255,127,0), 3)

        bboxes = model.detect(frame)
        
        for box in bboxes:

            x, y = box[:2]
            w, h = box[2] - x, box[3] - y

            valid_boxes = (w >= BOX_WIDTH_MIN and w <= BOX_WIDTH_MAX) and (h >= BOX_HEIGHT_MIN and h <= BOX_HEIGHT_MAX)

            if not valid_boxes:
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
            centroid = get_centroid(x, y, w, h)
            detections.append(centroid)
            cv2.circle(frame, centroid, 4, (0, 0,255), -1)

            for (x, y) in detections:
                if y < (line_y_position + OFFSET) and y > (line_y_position - OFFSET):
                    vehicle_count += 1
                    cv2.line(frame, (0, line_y_position), (image_width, line_y_position), (37, 29, 232), 3)
                    detections.remove((x, y))
                    print("Car detected : "+str(vehicle_count))

        cv2.putText(frame, "CROSSING VEHICLE COUNT : "+str(vehicle_count), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (37, 29, 232),5)
        cv2.imshow("Video" , frame)

        # Press Q on keyboard to  exit 
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
    
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main(video_path="sample/Night Time Traffic Camera video.mp4", line_y_position=550)