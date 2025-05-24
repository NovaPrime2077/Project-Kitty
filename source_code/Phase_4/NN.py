def main(Landmarks_folder,leg):
    import os
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, InputLayer, Masking, TimeDistributed
    from glob import glob
    from sklearn.model_selection import train_test_split


    def preprocess_pipeline(Landmarks_folder):
        """Enhanced preprocessing pipeline for pose landmark data"""
        
        # ====== Constants ======
        NUM_FRAMES = 120
        NUM_KEYPOINTS = 33  # MediaPipe provides 33 keypoints (0-32)
        LOWER_BODY_INDICES = list(range(23, 33))  # Keypoints 23-32 (10 keypoints)
        FEATURES_PER_KEYPOINT = 4  # x, y, z, visibility
        TOTAL_FEATURES = len(LOWER_BODY_INDICES) * FEATURES_PER_KEYPOINT  # Now 40

        def validate_and_fix_landmarks(landmarks):
            """Ensure consistent features even with missing data"""
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
                    print(f"{file_path} skipped: no valid frames")
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


        # ====== Main Processing ======
        print(f"Loading data from {Landmarks_folder}...")
        all_files = sorted([
            f for f in glob(os.path.join(Landmarks_folder, '*.npy'))
            if leg.lower() in os.path.basename(f).lower()
        ])

        processed_data = []
        
        skipped = 0
        for file in all_files:
            processed = load_and_process_file(file)
            if processed is not None:
                processed_data.append(processed)
            else:
                print(f"❌ Skipping file: {file}")
                skipped += 1

        print(f"\nTotal files: {len(all_files)}")
        print(f"Processed files: {len(processed_data)}")
        print(f"Skipped files: {skipped}")

        
        if not processed_data:
            raise ValueError("No valid training data found in the specified folder")
        
        X = np.array(processed_data)
        print(f"Successfully processed {X.shape[0]} sequences")
        print(f"Final data shape: {X.shape} (samples, frames, features)")
        
        # Train-validation split
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        
        return X_train, X_val
    # ====== Constants ======
    NUM_FRAMES = 120  # Fixed sequence length
    NUM_LOWER_BODY_KEYPOINTS = 10  # MediaPipe lower body indices (23–33)
    FEATURES_PER_KEYPOINT = 4  # (x, 0y, z, visibility)
    TOTAL_FEATURES = NUM_LOWER_BODY_KEYPOINTS * FEATURES_PER_KEYPOINT  # 40


    # ====== Load Data ======
    X_train, X_val = preprocess_pipeline(Landmarks_folder)
    print(X_train.shape)
    print(X_val.shape)
    
    print("Loaded", X_train.shape[0], "pose sequences.")

    # ====== Model Architecture ======
    model = Sequential([
        InputLayer(shape=(NUM_FRAMES, TOTAL_FEATURES)),  # (120, 40)

        Masking(mask_value=0.0),  # optional if you pad frames with zeros

        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        Dropout(0.3),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        Dropout(0.3),

        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),

        TimeDistributed(Dense(TOTAL_FEATURES))  # Predict pose per frame
    ])


    # Compile for sequence regression
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mean_squared_error'
    )

    model.summary()

    # ====== Train the Model ======
    model.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=20, batch_size=8)
    model.save(f"NovaPrime2077_{leg}.keras")  # Save the entire model
    print(f"NovaPrime2077_{leg}.keras")
# Example usage:
# predict_pose_similarity("test_landmarks/messi.npy")



    
