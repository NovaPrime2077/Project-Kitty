def main(all_players,video_path,file,landmark_path,player_id):
    import mediapipe as mp
    import numpy as np
    import cv2
    import os

    print(all_players.shape)
    frame_count, max_ids = all_players.shape[0], all_players.shape[1]
    print(frame_count,max_ids)
    player_bboxes = []
    for frame in range(frame_count):
        bbox = all_players[frame][player_id]
        if np.any(bbox):
            player_bboxes.append(bbox)
        else:
            player_bboxes.append([0, 0, 0, 0]) 

    player_bboxes = np.array(player_bboxes) 


    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False)

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise RuntimeError("Cannot open video")

    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    pose_results = [] 

    frame_idx = 0
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret or frame_idx >= len(player_bboxes):
            break

        l, t, r, b = player_bboxes[frame_idx]
        if l == 0 and t == 0 and r == 0 and b == 0:
            pose_results.append(None) 
            frame_idx += 1
            continue

        cropped = frame[t:b, l:r]
        if cropped.size == 0:
            print("Empty crop due to invalid bounding box:")
            continue
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)


        result = pose.process(cropped_rgb)

        if result.pose_landmarks:
            landmarks = []
            for lm in result.pose_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
            pose_results.append(landmarks)
        else:
            pose_results.append(None)

        frame_idx += 1


    pose_results = np.array(pose_results, dtype=object)
    landmark_output_path = os.path.join(landmark_path, f"{file}.npy")
    np.save(landmark_output_path, pose_results)
    print(f"Saved pose landmarks to player_{player_id}_pose_landmarks.npy")

    pose.close()
    vid.release()
    cv2.destroyAllWindows()


