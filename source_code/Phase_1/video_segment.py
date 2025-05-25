def main():
    import pandas as pd
    from moviepy.editor import VideoFileClip
    import os

    csv_path = r"\Project_Kitty_Pclub\data\CSV"
    raw_video_path = r"\Project_Kitty_Pclub\data\Raw videos"
    output_folder = r"\Project_Kitty_Pclub\data\Segmented clips"
    os.makedirs(output_folder, exist_ok=True)

    def second(s, e):
        m1, s1 = map(int, s.strip().split(":"))
        m2, s2 = map(int, e.strip().split(":"))
        start_sec = m1 * 60 + s1
        end_sec = m2 * 60 + s2
        return start_sec, end_sec

    for csv_file in os.listdir(csv_path):
        if not csv_file.endswith(".csv"):
            continue

        path_to_csv = os.path.join(csv_path, csv_file)
        data = pd.read_csv(path_to_csv)
        if "Penalties" in csv_file:
            category = "01_Penalties"
        elif "Goals" in csv_file:
            category = "03_Goals"
        elif "Free_Kicks" in csv_file:
            category = "02_Free_Kicks"
        else:
            continue

        vid_dir = os.path.join(raw_video_path, category)
        seg_dir = os.path.join(output_folder, category)
        os.makedirs(seg_dir, exist_ok=True)

        for i in range(len(data)):
            video_name = data.iloc[i, 0]
            video_path = os.path.join(vid_dir, f"{video_name}.mp4")

            if not os.path.exists(video_path):
                print(f"File not found: {video_path}")
                continue
            try:
                clip = VideoFileClip(video_path)
                start_time, end_time = second(data.iloc[i, 3], data.iloc[i, 4])  
                segment = clip.subclip(start_time, end_time)

                out_name = f"{i}_{data.iloc[i, 5]}.mp4"  
                out_path = os.path.join(seg_dir, out_name)

                segment.write_videofile(out_path, codec="libx264", audio_codec="aac")
                clip.close()
                segment.close()
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
