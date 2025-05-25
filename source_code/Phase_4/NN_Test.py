def main(Landmarks_folder):
    import os
    import numpy as np
    from glob import glob
    from tensorflow import keras 
    right = keras.models.load_model(r"\Project_Kitty_Pclub\source_code\NovaPrime2077_right.keras") #pretrained models
    left = keras.models.load_model(r"\Project_Kitty_Pclub\source_code\NovaPrime2077_left.keras")
    right.summary()
    left.summary()


    def preprocess_pipeline(Landmarks_folder):
        NUM_FRAMES = 120
        NUM_KEYPOINTS = 33  
        LOWER_BODY_INDICES = list(range(23, 33))  # Keypoints 23-32 (10 keypoints)
        FEATURES_PER_KEYPOINT = 4  # x, y, z, visibility
        TOTAL_FEATURES = len(LOWER_BODY_INDICES) * FEATURES_PER_KEYPOINT

        def validate_and_fix_landmarks(landmarks):
            if landmarks is None:
                return np.zeros((NUM_KEYPOINTS, FEATURES_PER_KEYPOINT))
            
            landmarks = np.array(landmarks)
            if landmarks.shape != (NUM_KEYPOINTS, FEATURES_PER_KEYPOINT):
                fixed_landmarks = np.zeros((NUM_KEYPOINTS, FEATURES_PER_KEYPOINT))
                valid_indices = min(landmarks.shape[0], NUM_KEYPOINTS)
                fixed_landmarks[:valid_indices] = landmarks[:valid_indices]
                return fixed_landmarks
            return landmarks

        def load_and_process_file(file_path):
            try:
                data = np.load(file_path, allow_pickle=True)
                valid_frames = [validate_and_fix_landmarks(f) for f in data if f is not None]

                if len(valid_frames) == 0:
                    print(f"{file_path} skipped: Since no valid frames")
                    return None

                sequence = np.array(valid_frames)[:, LOWER_BODY_INDICES, :]
                sequence[..., :3] = (sequence[..., :3] - sequence[..., :3].mean()) / sequence[..., :3].std()

                if sequence.shape[0] < NUM_FRAMES:
                    pad_len = NUM_FRAMES - sequence.shape[0]
                    sequence = np.pad(sequence, ((0, pad_len), (0, 0), (0, 0)), mode='edge')
                else:
                    sequence = sequence[:NUM_FRAMES]

                return sequence.reshape(NUM_FRAMES, -1)

            except Exception as e:
                print(f"{file_path} skipped due to error: {e}")
                return None


        print(f"Loading data from {Landmarks_folder}...")
        all_files = sorted(glob(os.path.join(Landmarks_folder, '*.npy')))


        processed_data = []
        
        skipped = 0
        for file in all_files:
            processed = load_and_process_file(file)
            if processed is not None:
                processed_data.append(processed)
            else:
                print(f"Skipping file: {file}")
                skipped += 1

        print(f"\nTotal files: {len(all_files)}")
        print(f"Processed files: {len(processed_data)}")
        print(f"Skipped files: {skipped}")

        
        if not processed_data:
            raise ValueError("Training data not present in the folder")
        
        X = np.array(processed_data)
        print(f"Successfully processed {X.shape[0]} sequences")
        print(f"Final data shape: {X.shape} (samples, frames, features)")
        
        
        return X 
    X_train = preprocess_pipeline(Landmarks_folder)
    print(X_train.shape)
    
    print("Pose Sequences have been Loaded ----------->", X_train.shape[0])
    right_val = right.predict(X_train)
    left_val = left.predict(X_train)

    return X_train,right_val, left_val



    
