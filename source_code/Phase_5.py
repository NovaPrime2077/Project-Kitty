import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from Phase_2 import detector_tracker
from Phase_3 import pose_tracker
from Phase_4 import NN_Test, visualize
import os
import cv2

# Paths (modify as needed)
TEST_VIDPATH = "D:/Project_Kitty_Pclub/test/Segmented clips/01_Penalties"
TEST_BOX_PATH = "D:/Project_Kitty_Pclub/test/Box_path/01_Penalties"
TEST_LANDMARK_PATH = "D:/Project_Kitty_Pclub/test/Landmarks/01_Penalties"

class PenaltyAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⚽ Penalty Kick Analyzer")
        
        # GUI Elements
        self.label = tk.Label(root, text="Upload a Penalty Video (MP4)")
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
        self.root.update()  # Force GUI update
        
        # Phase 2: Detect and track players (show OpenCV window)
        self.all_players = detector_tracker.main(
            self.video_path,
            os.path.splitext(os.path.basename(self.video_path))[0],
            TEST_BOX_PATH
        )
        
        # Ask for player ID after detection
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
            
            # Phase 3: Pose tracking for selected player
            pose_tracker.main(
                self.all_players,
                self.video_path,
                os.path.splitext(os.path.basename(self.video_path))[0],
                TEST_LANDMARK_PATH,
                player_id  # Pass the selected player ID
            )
            
            # Phase 4: NN prediction and visualization
            original_seq, pred_right, pred_left = NN_Test.main(TEST_LANDMARK_PATH)
            visualize.main(original_seq, pred_right, pred_left)
            
            messagebox.showinfo("Success", "Analysis complete! Check the visualization.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {str(e)}")
        
        self.status_label.config(text="Done!")

if __name__ == "__main__":
    root = tk.Tk()
    app = PenaltyAnalyzerApp(root)
    root.mainloop()