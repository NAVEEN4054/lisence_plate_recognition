import  cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model_dir = 'path_to_model_directory/saved_model'
model = tf.saved_model.load(model_dir)

# Function to run object detection
def detect_objects(image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    return detections

def draw_detections(image, detections):
    height, width, _ = image.shape
    for i in range(int(detections.pop('num_detections'))):
        box = detections['detection_boxes'][i].numpy()
        class_id = int(detections['detection_classes'][i].numpy())
        score = detections['detection_scores'][i].numpy()
        
        if score > 0.5:  # Threshold for considering a detection
            y_min, x_min, y_max, x_max = box
            (startX, startY, endX, endY) = (int(x_min * width), int(y_min * height),
                                            int(x_max * width), int(y_max * height))
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = f'ID: {class_id} Score: {score:.2f}'
            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    # Open a video file or a webcam stream
    cap = cv2.VideoCapture('input_video.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        detections = detect_objects(frame)

        # Draw detections on the frame
        draw_detections(frame, detections)

        # Display the frame
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
