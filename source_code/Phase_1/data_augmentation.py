def main(folder_path):
    import os
    import cv2

    def flip_videos_in_folder(folder_path):
        # List all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".mp4"):
                file_path = os.path.join(folder_path, filename)
                
                # Check if filename contains 'left' or 'right'
                if '_left' in filename.lower():
                    new_filename = filename.replace('_left', '_right')
                    flip_and_save(file_path, os.path.join(folder_path, new_filename))
                elif '_right' in filename.lower():
                    new_filename = filename.replace('_right', '_left')
                    flip_and_save(file_path, os.path.join(folder_path, new_filename))
                else:
                    print(f"Skipping {filename} (no 'left' or 'right' tag)")

    def flip_and_save(input_path, output_path):
        # Open the video file
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        
        # Create VideoWriter for the output
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip the frame horizontally (left ↔ right)
            flipped_frame = cv2.flip(frame, 1)
            out.write(flipped_frame)
        
        # Release resources
        cap.release()
        out.release()
        print(f"Saved flipped video: {output_path}")

    # Example usage
    flip_videos_in_folder(folder_path)