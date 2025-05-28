import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from Phase_2 import detector_tracker
from Phase_3 import pose_tracker
from Phase_4 import NN_Test, visualize
import os
import cv2
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
TEST_VIDPATH = os.path.join(project_root, "test", "Segmented_clips", "01_Penalties")
TEST_BOX_PATH = os.path.join(project_root, "test", "Box_path", "01_Penalties")
TEST_LANDMARK_PATH = os.path.join(project_root, "test", "Landmarks", "01_Penalties")

class PenaltyAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Penalty Kick Analyzer")
        

        self.label = tk.Label(root, text="Upload a Penalty Video (MP4), Recommended to be 2-3 seconds long")
        self.label.pack(pady=10)
        
        self.upload_btn = tk.Button(root, text="Browse Video", command=self.upload_video)
        self.upload_btn.pack(pady=5)
        
        self.analyze_btn = tk.Button(
            root, 
            text="Detect Players", 
            command=self.detect_players, 
            state=tk.DISABLED
        )
        self.analyze_btn.pack(pady=10)
        
        self.status_label = tk.Label(root, text="", fg="blue")
        self.status_label.pack(pady=10)
        
        self.video_path = ""
        self.all_players = None
    
    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if file_path:
            self.video_path = file_path
            self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}")
            self.analyze_btn.config(state=tk.NORMAL)
    
    def detect_players(self):
        if not self.video_path:
            messagebox.showerror("Error", "No video selected!")
            return
        
        self.status_label.config(text="Detecting players... (Check OpenCV window)")
        self.root.update()
        
        self.all_players = detector_tracker.main(
            self.video_path,
            os.path.splitext(os.path.basename(self.video_path))[0],
            TEST_BOX_PATH
        )
        player_id = simpledialog.askinteger(
            "Input", 
            "Enter the striker's Player ID (seen in OpenCV window):",
            parent=self.root,
            minvalue=0
        )
        if player_id is not None:
            self.analyze_player(player_id)
    
    def analyze_player(self, player_id):
        try:
            self.status_label.config(text="Extracting pose...")
            self.root.update()
            pose_tracker.main(
                self.all_players,
                self.video_path,
                os.path.splitext(os.path.basename(self.video_path))[0],
                TEST_LANDMARK_PATH,
                player_id 
            )
            original_seq, pred_right, pred_left = NN_Test.main(TEST_LANDMARK_PATH)
            total_right, total_left, strin, joints = visualize.main(original_seq, pred_right, pred_left)
            if total_right > total_left:
                messagebox.showinfo("Output", message= f"Since the Euclidean norm of left foot is lesser compared to right foot in frames 10-100 by{total_right-total_left}, the model predicts LEFT foot as the dominant one")
            else:
                messagebox.showinfo("Output",message=f"Since the Euclidean norm of right foot is lesser compared to left foot in frames 10-100 by {total_left-total_right}, the model predicts RIGHT foot as the dominant one")
            messagebox.showinfo("Final Evaluation" , message= f"{strin}, {joints}")
            lst = [TEST_VIDPATH, TEST_BOX_PATH, TEST_LANDMARK_PATH] 
            for directory in lst:  
                for filename in os.listdir(directory):  
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):  
                        try:
                            os.unlink(file_path) 
                            print("Evaluation",f"Deleted: {file_path}")  
                        except Exception as e:
                            print("Evauation", f"Failed to delete {file_path}: {e}")
        except Exception as e:
            lst = [TEST_VIDPATH, TEST_BOX_PATH, TEST_LANDMARK_PATH] 
            for directory in lst:  
                for filename in os.listdir(directory):  
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):  
                        try:
                            os.unlink(file_path) 
                            print(f"Deleted: {file_path}")  
                        except Exception as e:
                            print(f"Failed to delete {file_path}: {e}")
            messagebox.showerror("Error", f"Failed: {str(e)}")
        
        self.status_label.config(text="Done!")

if __name__ == "__main__":
    root = tk.Tk()
    app = PenaltyAnalyzerApp(root)
    root.mainloop()