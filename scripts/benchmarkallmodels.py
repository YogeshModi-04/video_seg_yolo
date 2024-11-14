import os
from ultralytics import YOLO
import logging
import cv2
import numpy as np
import time
import pandas as pd
import torch
import gc

# Set logging level for ultralytics
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)  # Reduce verbosity

def video_segmentation(input_video_path, output_video_path, model, class_names, target_class="bottle", report_path="benchmark_report.csv"):
    # Initialize benchmark metrics
    class_scores = []
    processing_start_time = time.time()  # Start time for processing (excluding video writing)
    total_frames = 0

    # Get the class index for the target class (e.g., "bottle")
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

        with torch.no_grad():
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
        # cv2.imshow("Segmented Video", segmented_frame)

        # Write the frame with segmentation overlays to the output video (excluded from processing time)
        out.write(segmented_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Clean up variables
        del results
        del segmented_frame
        del frame_scores
        torch.cuda.empty_cache()

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

def clear_cuda_cache():
    torch.cuda.empty_cache()
    with torch.cuda.device(0):
        torch.cuda.ipc_collect()
    gc.collect()

def benchmark_all_models(models_dir, input_video_path, report_base_dir, output_video_base_dir):
    # Create base directories for reports and videos
    os.makedirs(report_base_dir, exist_ok=True)
    os.makedirs(output_video_base_dir, exist_ok=True)
    
    # Get video duration, fps, and name
    cap = cv2.VideoCapture(input_video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames_in_video / video_fps if video_fps > 0 else 0
    cap.release()
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]

    # Include video name, duration, and FPS in the directory name
    video_info_dir = f"{video_name}_{int(video_duration)} seconds_{video_fps}fps"
    
    for model_type in ["onnx", "TensorRT", "Pt"]:
        model_type_dir = os.path.join(models_dir, model_type)
        if not os.path.isdir(model_type_dir):
            continue  # Skip if model type directory does not exist

        for model_file in os.listdir(model_type_dir):
            model_path = os.path.join(model_type_dir, model_file)
            if not os.path.isfile(model_path):
                continue  # Skip if it's not a file

            # Set paths for saving output video and report
            model_name, _ = os.path.splitext(model_file)
            output_video_dir = os.path.join(output_video_base_dir, video_info_dir, model_type)
            os.makedirs(output_video_dir, exist_ok=True)
            output_video_path = os.path.join(output_video_dir, f"{model_name}.mp4")

            report_dir = os.path.join(report_base_dir, video_info_dir, model_type)
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, f"{model_name}.csv")

            try:
                # Load model based on file type
                print(f"Benchmarking model: {model_name}")

                # Clear cache before loading model
                clear_cuda_cache()

                if model_type == "Pt":
                    # Load .pt model with CUDA if available
                    model = YOLO(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
                else:
                    # Load .onnx or .engine model without device setting
                    model = YOLO(model_path)

                # Define default class names if not available
                class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

                # Run video segmentation and benchmarking
                video_segmentation(input_video_path, output_video_path, model, class_names, report_path=report_path)

                # Clear GPU memory after processing each model
                del model
                clear_cuda_cache()
                gc.collect()
                print(f"Cleared GPU memory after benchmarking model: {model_name}")

            except Exception as e:
                print(f"Error processing model {model_name}: {e}")
                del model
                clear_cuda_cache()
                gc.collect()
                continue

# Example usage
models_dir = "./models"
input_video_path = "test_5.mp4"
report_base_dir = "./report"
output_video_base_dir = "./segmented_videos"

benchmark_all_models(models_dir, input_video_path, report_base_dir, output_video_base_dir)
