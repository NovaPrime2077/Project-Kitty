def main():
    import pandas as pd
    import yt_dlp
    import os
    import glob
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current script directory (script_dir): {script_dir}")
    one_level_higher = os.path.dirname(script_dir)
    print(f"One level higher from script_dir: {one_level_higher}")
    project_root = os.path.dirname(one_level_higher)
    print(f"Project root: {project_root}")
    
    folder_path = os.path.join(project_root, "data", "CSV")
    base_output_dir = os.path.join(project_root, "data", "Raw_videos")


    print("Project root is:", project_root)
    for filename in os.listdir(folder_path):
        if not filename.endswith('.csv'):
            continue

        file_path = os.path.join(folder_path, filename)
        data = pd.read_csv(file_path)

        if filename == "01_Penalties.csv":
            output_dir = os.path.join(base_output_dir, "01_Penalties")
        elif filename == "02_Free_Kicks.csv":
            output_dir = os.path.join(base_output_dir, "02_Free_Kicks")
        else:
            output_dir = os.path.join(base_output_dir, "03_Goals")

        os.makedirs(output_dir, exist_ok=True)

        ydl_opts = {
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'format': 'bestvideo+bestaudio',
            'merge_output_format': 'mp4',
        }

        video_names = []
        seen = set()

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for i, video_url in enumerate(data.iloc[:, 1]):
                if pd.isna(video_url) or video_url in seen:
                    continue
                seen.add(video_url)
                try:
                    ydl.download([video_url])
                    video_names.append(data.iloc[i, 0])
                except Exception as e:
                    print(f"[ERROR] Failed to download: {video_url}\nReason: {e}")

        downloaded_files = glob.glob(os.path.join(output_dir, '*.mp4'))
        downloaded_files.sort(key=os.path.getmtime)

        for old_path, video_name in zip(downloaded_files, video_names):
            safe_name = "".join(c for c in str(video_name) if c not in r'<>:"/\|?*')
            new_path = os.path.join(output_dir, f"{safe_name}.mp4")
            try:
                os.rename(old_path, new_path)
            except FileExistsError:
                print(f"[SKIPPED] File already exists: {new_path}")
            except Exception as e:
                print(f"[ERROR] Could not rename '{old_path}' to '{new_path}': {e}")
