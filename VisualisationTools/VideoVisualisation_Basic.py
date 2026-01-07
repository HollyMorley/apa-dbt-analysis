import cv2

def main():
    video_file = r"C:\Users\hmorl\Documents\HM_20230316_APACharExt_FAA-1035246_LR_side_1.avi" # Path to your AVI video file
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    cv2.namedWindow('Video Player', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Video Player', 800, 600)  # Adjust window size as needed

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("End of video reached.")
            break

        # Add label for current frame number
        cv2.putText(frame, f"Frame: {frame_number}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Video Player', frame)
        instructions = "Instructions:\nPress 'n' to move to the next frame\nPress 'p' to move to the previous frame\nPress 's' to skip to a specific frame\nPress 'q' to quit"
        cv2.putText(frame, instructions, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):  # Quit if 'q' is pressed
            break
        elif key == ord('n'):  # Move to the next frame if 'n' is pressed
            frame_number += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        elif key == ord('p'):  # Move to the previous frame if 'p' is pressed
            if frame_number > 0:
                frame_number -= 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        elif key == ord('f'):  # fast forward 500 frames 'f'' is pressed
            if frame_number > 0:
                frame_number += 100
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        elif key == ord('r'):  # rewind 500 frames 'f'' is pressed
            if frame_number > 0:
                frame_number -= 100
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        elif key == ord('s'):  # Skip to a specific frame number if 's' is pressed
            target_frame = int(input("Enter the frame number to skip to: "))
            frame_number = target_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
