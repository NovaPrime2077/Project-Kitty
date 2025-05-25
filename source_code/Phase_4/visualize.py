def main(org_seq, pred_seq_right, pred_seq_left):
    import numpy as np
    import matplotlib.pyplot as plt

    def plot_pose_deviation(original_sequence, predicted_sequence_right, predicted_sequence_left):
        original_sequence = np.squeeze(original_sequence)
        predicted_sequence_right = np.squeeze(predicted_sequence_right)
        predicted_sequence_left = np.squeeze(predicted_sequence_left)

        assert original_sequence.shape == (120, 40), f"Expected shape (120, 40), got {original_sequence.shape}"
        assert predicted_sequence_right.shape == (120, 40)
        assert predicted_sequence_left.shape == (120, 40)

        difference_right = np.linalg.norm(original_sequence - predicted_sequence_right, axis=1)
        difference_left = np.linalg.norm(original_sequence - predicted_sequence_left, axis=1)
        total_right = np.sum(difference_right)
        total_left = np.sum(difference_left)

        plt.figure(figsize=(12, 6))
        plt.plot(range(1, 121), difference_right, color='blue', linewidth=2, label='Right Prediction')
        plt.plot(range(1, 121), difference_left, color='green', linewidth=2, label='Left Prediction')
        top_3_right = np.argsort(difference_right)[-3:] + 1
        top_3_left = np.argsort(difference_left)[-3:] + 1
        for idx in top_3_right:
            plt.axvline(x=idx, color='blue', linestyle='--', alpha=0.3)
        for idx in top_3_left:
            plt.axvline(x=idx, color='green', linestyle='--', alpha=0.3)

        plt.title("Deviation from Professional Pose Across Frames")
        plt.xlabel("Frame")
        plt.ylabel("Euclidean Norm")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return total_right, total_left

    total_right, total_left = plot_pose_deviation(org_seq, pred_seq_right, pred_seq_left)
    print(f"Total deviation (Right): {total_right:.4f}")
    print(f"Total deviation (Left): {total_left:.4f}")
    if(total_right>total_left):
        print(f"Since the Euclidean norm of left foot is lesser compared to right foot, the model predicts LEFT foot as the dominant one")
    else:
        print(f"Since the Euclidean norm of right foot is lesser compared to left foot, the model predicts RIGHT foot as the dominant one")
    return total_right, total_left