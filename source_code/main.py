from Phase_1 import download, video_segment,data_augmentation
from Phase_2 import detector_tracker
from Phase_3 import pose_tracker
from Phase_4 import NN
import os

def main():
    vidpath = "\Project_Kitty_Pclub\data\Segmented clips\\01_Penalties"
    vidpath = vidpath.replace("\\","/")
    boxx_path = "\Project_Kitty_Pclub\data\Box_path\\01_Penalties"
    boxx_path = boxx_path.replace("\\","/")
    landmark_path = "\Project_Kitty_Pclub\data\Landmarks\\01_Penalties"
    landmark_path = landmark_path.replace("\\","/")
    download.main()
    video_segment.main()
    data_augmentation.main("/Project_Kitty_Pclub/data/Segmented clips/01_Penalties")
    for file in os.listdir(vidpath):
        if not file.endswith(".mp4"):
            continue
        video_path = os.path.join(vidpath, file).replace("\\", "/")
        all_players = detector_tracker.main(video_path, file.removesuffix(".mp4"), boxx_path)
        player_id = int(input())
        pose_tracker.main(all_players, video_path,file.removesuffix(".mp4"),landmark_path,player_id)
    NN.main(landmark_path,"left")
    NN.main(landmark_path,"right")

if __name__ == "__main__":
    main()
