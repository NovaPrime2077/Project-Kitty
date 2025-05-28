from Phase_1 import download, video_segment,data_augmentation
from Phase_2 import detector_tracker
from Phase_3 import pose_tracker
from Phase_4 import NN
import os

def main():

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(SCRIPT_DIR)
    vidpath = os.path.join(project_root,"data","Segmented_clips","01_Penalties")
    boxx_path = os.path.join(project_root,"data","Box_path","01_Penalties")
    landmark_path = os.path.join(project_root,"data","Landmarks","01_Penalties")
    # download.main() # I recommend commenting these and using Google drive to download the segmented or raw data to save time
    # video_segment.main() # I recommend commenting these and using Google drive to download the segmented or raw data to save time
    # data_augmentation.main(os.path.join(project_root,"data","Segmented_clips","01_Penalties")) # I recommend commenting these and using Google drive to download the segmented or raw data to save time
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
