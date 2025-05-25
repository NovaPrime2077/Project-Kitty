# Project Kitty
The Project aims to develop an algorithmm which can through the power of Machine Learning and Computer Vision can take sports analytics light years ahead from where it is now. The current model touches on it by analyzing how professionals take their penalties and this can help enthusiasts in these sports to learn faster and better. 

First, I would like to link the provide a few external links that are quite essential in running the model on your local machine. Here are a few links to go through before you hop on the repo, please do read them they will surely make your life easier in the long run !!


link for raw and segmented videos - https://drive.google.com/drive/folders/1TfuGtzZh91Fp52pifOfZexJExU5jjpVg?usp=drive_link

link for the detailed report on the model - https://github.com/NovaPrime2077/Project-Kitty/blob/main/Report.pdf

link of how to run **Phase_5.py** - https://drive.google.com/file/d/1Z13ap0JrdWhW26oNtDBuQ1ljHtAXRFGy/view?usp=drive_link
## Basic Structure of the Repo 
Here is the basic structure of the repo, please note to follow the exact structure otherwise programs might fail to run on your local machine.
```
Project_Kitty_Pclub/
├── data/                        Raw and processed video data exists here
│   ├── Box_path/01_Penalties/
│   ├── CSV/
│   ├── Landmarks/01_Penalties/
│   ├── Raw videos/01_Penalties/
│   └── Segmented clips/01_Penalties/
├── source_code/                 Core codebase split by development phases
│   ├── Phase_1/                 Download, Segmentation and Augmentation 
│   ├── Phase_2/                 YOLO + DEEPSORT
│   ├── Phase_3/                 MediaPipe Pose tracking
│   ├── Phase_4/                 CNN & LSTM + Visualisation
│   ├── Phase_5.py               Tkinter File to test model in one click
│   ├── main.py                  Training Script
│   ├── main_test.py             Testing Script
│   ├── NovaPrime2077_left.keras    Trained Model for left foot
│   ├── NovaPrime2077_right.keras   Trained Model for right foot
│   └── yolo11m.pt               YOLOv11 weights
├── test/                        Test data organized like main data/
│   ├── Box_path/01_Penalties/
│   ├── Landmarks/01_Penalties/
│   └── Segmented clips/01_Penalties/
├── report.pdf                   Full technical report
├── LICENSE                      MIT License
└── README.md                    You are here !!
```
There are a few warnings that I wanted to give before we jump in, please bear with me:
1. **main.py** - This file is the master file it contains all the code block from Phase 1 to 4 i.e. therefore it will download, segment, augment, use YOLO, DeepSort, Mediapipe and CNN & LSTM. **I recommend commenting the downloading, segmentation and augmentation part of the file** this is because not only does these process take up ton of time they are of no value to the current project except for data procuring so I recommend **using the google drive link to download these data** and directly use the model's main tasks like player detection, tracking,etc.
2. **Phase_5.py** - I highly recommend using this file for **viewing purposes** since this file doesn't require any knowledge and technicality of the repo.
**Please follow the video linked above to know how to run Phase_5.py**
3. **main_test.py** - for testing the model from inside this file can also be used but I recommend using Phase_5.py instead.  
4. **detector_tracker.py** - The file uses YOLOV11 and the model runs using GPU i.e. CUDA framework, I highly recommend using the feature of YOLO if you have a CUDA supported device and are going to run the main.py file.  
5. It is very important to note that **moviepy's current version has tons of bugs** that's why certain features won't work, I recommend using Version - 1.0.3 to avoid any problems in running and analyzing the model if main.py will be used. 

## Requirements
The following components are required to run **OffSide** on your native machine:
1. Python
2. YOLO11 from Ultralytics
3. DeepSort from deep_sort_realtime
4. Tensorflow
5. Streamlit
6. NumPy
7. Pandas
8. yt-dlp
9. moviepy

Please note that you don't really need Segmented clips and raw videos to run any file **except main.py** because the repo already contains Landmarks and Boxes .npy files to ensure that model runs perfectly. 
That's all, I wish you will like **OffSide**. Thank you for your patience. 
### Author
*****NovaPrime2077***** 


