from ultralytics import YOLO

# Load a pretrained YOLOv8 model (e.g., yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
model = YOLO('/Users/jamesemilian/triage/equipleth/yolo_chkpts/yolov8n-face.pt')  # You can replace 'yolov8n.pt' with other model sizes

# Perform inference on an image
results = model('/Users/jamesemilian/triage/equipleth/experiment_data/M24-04-fusion-test-data-60s-nomcap/output_5_60s/frames/iphone_rear/1716563771563198804.jpg')  # Replace with your image path

# Print results
# results.print()

# Optionally, display the image with detections
# results.show()

# Accessing the bounding boxes, confidence scores, and class labels
for result in results:
    print(f'Result is: {result}')
    boxes = result.boxes  # Bounding boxes
    probs = result.probs
    print(f'Result prob is: {probs}')
    # scores = result.scores  # Confidence scores
    # labels = result.labels  # Class labels
    # for box, score, label in zip(boxes, scores, labels):
    #     print(f'Box: {box}, Score: {score}, Label: {label}')
    result.show()
    # Save results to a file (if needed)
    result.save('/Users/jamesemilian/triage/equipleth/experiment_data/face_detected_image.jpg')  # Save the result image
