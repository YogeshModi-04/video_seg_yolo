import os
from ultralytics import YOLO
import logging
import cv2
import numpy as np
import time
import pandas as pd

# Set logging level for ultralytics
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)  # Reduce verbosity

def video_segmentation(input_video_path, output_video_path, model, target_class="bottle", report_path="benchmark_report.csv"):
    # Initialize benchmark metrics
    class_scores = []
    processing_start_time = time.time()  # Start time for processing (excluding video writing)
    total_frames = 0

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
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (width, height))

    # Calculate the duration of the video in seconds
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames_in_video / video_fps if video_fps > 0 else 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Measure start time for the frame processing only
        frame_start_time = time.time()

        # Perform segmentation on the frame
        results = model(frame)

        # Copy the original frame to overlay segmentation results
        segmented_frame = frame.copy()

        frame_scores = []  # Store scores for target class in this frame

        for result in results:
            if result.masks is not None:
                # Only consider masks that match the target class index
                masks = result.masks.xyn  # Normalized coordinates for masks
                classes = result.boxes.cls  # Class IDs for detected objects
                scores = result.boxes.conf  # Confidence scores

                for mask, cls, score in zip(masks, classes, scores):
                    # Check if the detected class is the target class (e.g., "bottle")
                    if int(cls) == target_class_idx:
                        # Append the score to frame_scores after moving to CPU
                        frame_scores.append(score.cpu().numpy())

                        # Convert normalized mask coordinates to absolute frame coordinates
                        absolute_mask = (mask * [width, height]).astype(np.int32)

                        # Check if mask points are in the right shape (N, 2) before passing to fillPoly
                        if absolute_mask.shape[1] == 2 and absolute_mask.ndim == 2:
                            # Draw the polygon mask on the frame with a chosen color
                            color = (0, 0, 255)  # Red color for bottle segments
                            cv2.fillPoly(segmented_frame, [absolute_mask], color=color)

        # Append the average confidence score of this frame to class_scores (or 0 if no detections)
        avg_score = np.mean(frame_scores) if frame_scores else 0
        class_scores.append(avg_score)

        # Increment frame count
        total_frames += 1

        # Measure frame processing time without including video writing time
        frame_end_time = time.time()
        frame_processing_time = frame_end_time - frame_start_time

        # Display the segmented frame
        cv2.imshow("Segmented Video", segmented_frame)

        # Write the frame with segmentation overlays to the output video (excluded from processing time)
        out.write(segmented_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Calculate benchmarking metrics
    processing_end_time = time.time()
    total_processing_time = processing_end_time - processing_start_time  # Only processing time
    average_frame_processing_time = total_processing_time / total_frames if total_frames > 0 else 0
    average_target_class_score = np.mean(class_scores) * 100  # Convert to percentage
    processing_fps = total_frames / total_processing_time if total_processing_time > 0 else 0

    # Print benchmarking results
    print(f"Total time: {total_processing_time:.2f} sec")
    print(f"Video FPS: {video_fps} fps")
    print(f"Video Duration: {video_duration:.2f} sec")
    print(f"Average Frame Processing Time: {average_frame_processing_time:.4f} sec")
    print(f"Target Class Score: {average_target_class_score:.2f}%")
    print(f"Processing FPS: {processing_fps:.2f} fps")

    # Save benchmark results to a CSV report
    report_data = {
        "Metric": [
            "Total time", 
            "Video FPS", 
            "Video Duration",
            "Average Frame Processing Time", 
            "Target Class Score", 
            "Processing FPS"
        ],
        "Value": [
            f"{total_processing_time:.2f} sec",
            f"{video_fps} fps",
            f"{video_duration:.2f} sec",
            f"{average_frame_processing_time:.4f} sec",
            f"{average_target_class_score:.2f}%",
            f"{processing_fps:.2f} fps"
        ]
    }
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(report_path, index=False)

    print(f"Benchmark report saved to {report_path}")

# Example usage
input_video_path = "test.mp4"  # Path to your input video file
output_video_path = "segmented_output_bottle.mp4"  # Path to save the output segmented video
model_path = "yolov8n-seg.pt"  # Model path
report_path = "benchmark_report.csv"  # CSV report file

# Load the YOLO model
model = YOLO(model_path)

# Run segmentation and benchmarking
video_segmentation(input_video_path, output_video_path, model, target_class="bottle", report_path=report_path)
