def main(org_seq, pred_seq_right, pred_seq_left):
    import numpy as np
    import matplotlib
    matplotlib.use('TkAgg')  #Found that on linux conflict is arising for matplotlib due to to backend incompatibility
    import matplotlib.pyplot as plt


    KEY_POINTS = {
        23: "left hip",
        24: "right hip",
        25: "left knee",
        26: "right knee",
        27: "left ankle",
        28: "right ankle",
        29: "left heel",
        30: "right heel",
        31: "left foot index",
        32: "right foot index"
    }

    def plot_pose_deviation(original_sequence, predicted_sequence_right, predicted_sequence_left):
        original_sequence = np.squeeze(original_sequence)
        predicted_sequence_right = np.squeeze(predicted_sequence_right)
        predicted_sequence_left = np.squeeze(predicted_sequence_left)

        assert original_sequence.shape == (120, 40), f"Expected shape (120, 40), got {original_sequence.shape}"
        assert predicted_sequence_right.shape == (120, 40)
        assert predicted_sequence_left.shape == (120, 40)
        frame_start, frame_end = 10, 100
        original_subset = original_sequence[frame_start-1:frame_end]
        predicted_right_subset = predicted_sequence_right[frame_start-1:frame_end]
        predicted_left_subset = predicted_sequence_left[frame_start-1:frame_end]

        joint_diffs_right = original_subset - predicted_right_subset
        joint_diffs_left = original_subset - predicted_left_subset
        
        difference_right = np.linalg.norm(joint_diffs_right, axis=1)
        difference_left = np.linalg.norm(joint_diffs_left, axis=1)
        total_right = np.sum(difference_right)
        total_left = np.sum(difference_left)

        avg_joint_diff_right = np.mean(np.abs(joint_diffs_right), axis=0)
        avg_joint_diff_left = np.mean(np.abs(joint_diffs_left), axis=0)

        top_joints_right = np.argsort(avg_joint_diff_right)[-3:][::-1] % len(KEY_POINTS)
        top_joints_left = np.argsort(avg_joint_diff_left)[-3:][::-1] % len(KEY_POINTS)

        plt.figure(figsize=(12, 6))

        plt.plot(range(1, 121), np.linalg.norm(original_sequence - predicted_sequence_right, axis=1), 
                color='blue', linewidth=2, label='Right Prediction')
        plt.plot(range(1, 121), np.linalg.norm(original_sequence - predicted_sequence_left, axis=1), 
                color='green', linewidth=2, label='Left Prediction')
        

        plt.axvspan(frame_start, frame_end, color='yellow', alpha=0.2, label='Analyzed Frames (10-100)')
        

        subset_top_3_right = np.argsort(difference_right)[-3:] + frame_start
        subset_top_3_left = np.argsort(difference_left)[-3:] + frame_start
        for idx in subset_top_3_right:
            plt.axvline(x=idx, color='blue', linestyle='--', alpha=0.3)
        for idx in subset_top_3_left:
            plt.axvline(x=idx, color='green', linestyle='--', alpha=0.3)

        plt.title(f"Deviation from Professional Pose (Analyzing Frames {frame_start}-{frame_end})")
        plt.xlabel("Frame")
        plt.ylabel("Euclidean Norm")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return total_right, total_left, top_joints_right, top_joints_left

    total_right, total_left, top_joints_right, top_joints_left = plot_pose_deviation(org_seq, pred_seq_right, pred_seq_left)
    print(f"Total deviation (Right) for frames 10-100: {total_right:.4f}")
    print(f"Total deviation (Left) for frames 10-100: {total_left:.4f}")
    
    if total_right > total_left:
        # print(f"Since the Euclidean norm of left foot is lesser compared to right foot in frames 10-100, the model predicts LEFT foot as the dominant one")
        smaller_norm = total_left
        problematic_joints = top_joints_left
    else:
        # print(f"Since the Euclidean norm of right foot is lesser compared to left foot in frames 10-100, the model predicts RIGHT foot as the dominant one")
        smaller_norm = total_right
        problematic_joints = top_joints_right
    
    if smaller_norm > 300:
        valid_problematic_joints = [j for j in problematic_joints if j in KEY_POINTS]

        if valid_problematic_joints:
            str1 = "You can do much better than this, just focus on the joints below:"
            # print("\nYou can do much better than this, just focus on the joints below:")
            # for joint_idx in valid_problematic_joints:
            #     print(f"- {KEY_POINTS[joint_idx]}")
            return total_right, total_left, str2,valid_problematic_joints
        else:
            str1 = "The overall form is wrong, you need to practice harder!!"
            # print("\nThe overall form is wrong, you need to practice harder!!")
        
            return total_right, total_left,str1,""
    else:

        valid_problematic_joints = [j for j in problematic_joints if j in KEY_POINTS]

        if valid_problematic_joints:
            str2 = "You are doing good but can do better, focus on:"
            # print("\nYou are doing good but can do better, focus on:")
            # for joint_idx in valid_problematic_joints:
            #     print(f"- {KEY_POINTS[joint_idx]}")
            return total_right, total_left, str2,valid_problematic_joints
        else:
            str2 = "Good form overall! No major joint deviations detected."
            # print("\nGood form overall! No major joint deviations detected.")
            return total_right, total_left, str2, "Congrats!!"