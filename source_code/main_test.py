from tensorflow import keras 
from Phase_2 import detector_tracker
from Phase_3 import pose_tracker
from Phase_4 import NN_Test, visualize
import os

def main():
    test_vidpath = "\Project_Kitty_Pclub\\test\Segmented clips\\01_Penalties"
    test_vidpath = test_vidpath.replace("\\","/")
    
    test_boxx_path = "\Project_Kitty_Pclub\\test\Box_path\\01_Penalties"
    test_boxx_path = test_boxx_path.replace("\\","/")
   
    test_landmark_path = "\Project_Kitty_Pclub\\test\Landmarks\\01_Penalties"
    test_landmark_path = test_landmark_path.replace("\\","/")
    
    for file in os.listdir(test_vidpath):
        if not file.endswith(".mp4"):
            continue
        video_path = os.path.join(test_vidpath, file).replace("\\", "/")
        all_players = detector_tracker.main(video_path, file.removesuffix(".mp4"), test_boxx_path)
        player_id = int(input())
        pose_tracker.main(all_players, video_path,file.removesuffix(".mp4"),test_landmark_path, player_id)
    original_seq, predicted_right_seq, predicted_left_seq = NN_Test.main(test_landmark_path)
    visualize.main(original_seq, predicted_right_seq,predicted_left_seq)
    lst = [test_vidpath, test_boxx_path, test_landmark_path] 

    for directory in lst:  
        for filename in os.listdir(directory):  
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):  
                try:
                    os.unlink(file_path) 
                    print(f"Deleted: {file_path}")  
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
if __name__ == "__main__":
    main()



