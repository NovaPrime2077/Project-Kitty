def main(video_path, file, boxx_path):
    from ultralytics import YOLO
    import cv2
    from deep_sort_realtime.deepsort_tracker import DeepSort
    import numpy as np
    import os

    model = YOLO("yolo11m.pt")
    tracker = DeepSort(max_age=45)

    os.makedirs(boxx_path, exist_ok=True)

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    max_ids = 10 

    all_bboxes = np.zeros((frame_count, max_ids, 4), dtype=np.int32)
    frame_index = 0

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret or frame_index >= frame_count:
            break

        results = model(frame, conf=0.5, device='cuda')[0] #CUDA needed for faster running of YOLO

        detections = []
        for box in results.boxes:
            if int(box.cls[0]) != 0:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = int(track.track_id)
            l, t, r, b = map(int, track.to_ltrb())
            if tid < max_ids:
                all_bboxes[frame_index, tid] = [l, t, r, b]
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{tid}", (l, t - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Track All Players (Press q to quit)", frame)
        if cv2.waitKey(90) & 0xFF == ord('q'):
            break

        frame_index += 1

    vid.release()
    cv2.destroyAllWindows()

    bbox_output_path = os.path.join(boxx_path, f"{file}.npy")
    np.save(bbox_output_path, all_bboxes)
    print(f"Saved all bounding boxes to: {bbox_output_path}")

    return all_bboxes
