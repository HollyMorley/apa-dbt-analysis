"""Extracts and saves individual frames from video files."""
import cv2
import argparse

"""
To use: 'python script_name.py input_video.mp4 100 output_image.jpg'
"""

def save_frame_as_image(video_path, frame_number, output_path):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not video_capture.isOpened():
        print("Error: Couldn't open the video file.")
        return

    # Set the frame number to the desired frame
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = video_capture.read()

    # Check if the frame is read successfully
    if not ret:
        print("Error: Couldn't read frame.")
        return

    # Save the frame as a JPEG image
    cv2.imwrite(output_path, frame)

    # Release the video capture object
    video_capture.release()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Save a frame from a video file as a JPEG image')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('frame_number', type=int, help='Frame number to save as an image')
    parser.add_argument('output_path', type=str, help='Path to save the output image')
    args = parser.parse_args()

    # Call the function to save the frame as an image
    save_frame_as_image(args.video_path, args.frame_number, args.output_path)

if __name__ == "__main__":
    main()
