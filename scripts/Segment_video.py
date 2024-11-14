import os
from ultralytics import YOLO
import logging
import cv2
import numpy as np
# Set logging level for ultralytics
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)  # Reduce the verbosity

model = YOLO("/home/chichi/code/segmentation/models/Pt/yolo11n-seg.pt")
print("Classes: ",model.names)

def video_segmentation(input_video_path, output_video_path,model, target_class="bottle"):
    # Load the YOLO segmentation model
    # model = YOLO(model_name)

    # Get the class index for the target class (e.g., "bottle")
    class_names = model.names
    target_class_idx = next((idx for idx, name in class_names.items() if name == target_class), None)
    if target_class_idx is None:
        print(f"Class '{target_class}' not found in model classes.")
        return

    # Initialize the video capture and writer
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform segmentation on the frame
        results = model(frame)

        # Copy the original frame to overlay segmentation results
        segmented_frame = frame.copy()

        for result in results:
            if result.masks is not None:
                # Only consider masks that match the target class index
                masks = result.masks.xyn  # Normalized coordinates for masks
                classes = result.boxes.cls  # Class IDs for detected objects

                for mask, cls in zip(masks, classes):
                    # Check if the detected class is the target class (e.g., "bottle")
                    if int(cls) == target_class_idx:
                        # Convert normalized mask coordinates to absolute frame coordinates
                        absolute_mask = (mask * [width, height]).astype(np.int32)

                        # Check if mask points are in the right shape (N, 2) before passing to fillPoly
                        if absolute_mask.shape[1] == 2 and absolute_mask.ndim == 2:
                            # Draw the polygon mask on the frame with a chosen color
                            color = (0, 0, 255)  # Red color for bottle segments
                            cv2.fillPoly(segmented_frame, [absolute_mask], color=color)

        # Display the segmented frame
        cv2.imshow("Segmented Video", segmented_frame)

        # Write the frame with segmentation overlays
        out.write(segmented_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Segmentation video saved successfully for class:", target_class)

# Example usage
input_video_path = "test.mp4"  # Path to your input video file
output_video_path = "yolov8nengine.mp4"  # Path to save the output segmented video
video_segmentation(input_video_path, output_video_path, model,target_class="bottle")
