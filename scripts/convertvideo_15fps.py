from moviepy.editor import VideoFileClip

def reduce_fps(input_file, output_file):
    # Load the video
    clip = VideoFileClip(input_file)

    # Set the new frame rate to 15 fps
    clip = clip.set_fps(15)

    # Write the result to the output file
    clip.write_videofile(output_file, fps=15)

if __name__ == "__main__":
    # Specify the paths to your input and output videos here
    input_file = "/home/chichi/code/segmentation/test_1.mp4"   # Replace with your input video path
    output_file = "/home/chichi/code/segmentation/test_4.mp4" # Replace with your output video path

    reduce_fps(input_file, output_file)
