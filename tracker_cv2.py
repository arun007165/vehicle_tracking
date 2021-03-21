
# Imports
import cv2
import numpy as np
import time


# Minimum Box Dimensions
BOX_WIDTH_MIN = 120
BOX_HEIGHT_MIN = 80

BOX_WIDTH_MAX = 1200
BOX_HEIGHT_MAX = 800

OFFSET = 8
FPS_DELAY = 60

	
def get_centroid(x, y, w, h):
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    return cx, cy
    

def main(video_path, line_y_position):

    cap = cv2.VideoCapture(video_path)
    object_detector = cv2.bgsegm.createBackgroundSubtractorMOG()

    vehicle_count= 0
    detections = []

    while True:
        
        ret, frame = cap.read()

        image_height, image_width, _ = frame.shape
        if not ret:
            break

        interval = float(1 / FPS_DELAY)
        time.sleep(interval)

        # Convert to Gray Scale
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur
        frame_blur = cv2.GaussianBlur(frame_gray, (3, 3), 5)

        # Apply Object Detector
        frame_subtracted = object_detector.apply(frame_blur)
        frame_dilated = cv2.dilate(frame_subtracted,np.ones((5,5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        frame_morph = cv2.morphologyEx(frame_dilated, cv2. MORPH_CLOSE , kernel)
        contours, h = cv2.findContours(frame_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.line(frame, (0, line_y_position), (image_width, line_y_position), (255,127,0), 3)
        
        for c in contours:

            (x, y, w, h) = cv2.boundingRect(c)

            valid_contours = (w >= BOX_WIDTH_MIN and w <= BOX_WIDTH_MAX) and (h >= BOX_HEIGHT_MIN and h <= BOX_HEIGHT_MAX)

            if not valid_contours:
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
        cv2.imshow("Videoabc" , frame_dilated)

        # Press Q on keyboard to  exit 
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
    
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main(video_path="sample/Night Time Traffic Camera video.mp4", line_y_position=550)