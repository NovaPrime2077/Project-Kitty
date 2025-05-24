def main(org_seq, pred_seq_right, pred_seq_left):
    import numpy as np
    import matplotlib.pyplot as plt

    def plot_pose_deviation(original_sequence, predicted_sequence_right, predicted_sequence_left):
        """
        Plots the Euclidean distance (L2 norm) between original and predicted pose
        vectors for each frame (total 120 frames).

        Args:
            original_sequence: np.array of shape (120, 40) or (1, 120, 40)
            predicted_sequence_right: np.array of shape (120, 40) or (1, 120, 40)
            predicted_sequence_left: np.array of shape (120, 40) or (1, 120, 40)
        """
        # Squeeze in case shape is (1, 120, 40)
        original_sequence = np.squeeze(original_sequence)
        predicted_sequence_right = np.squeeze(predicted_sequence_right)
        predicted_sequence_left = np.squeeze(predicted_sequence_left)

        # Safety check
        assert original_sequence.shape == (120, 40), f"Expected shape (120, 40), got {original_sequence.shape}"
        assert predicted_sequence_right.shape == (120, 40)
        assert predicted_sequence_left.shape == (120, 40)

        # Compute Euclidean norm across features for each frame
        difference_right = np.linalg.norm(original_sequence - predicted_sequence_right, axis=1)
        difference_left = np.linalg.norm(original_sequence - predicted_sequence_left, axis=1)

        # Plot both on the same graph
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, 121), difference_right, color='blue', linewidth=2, label='Right Prediction')
        plt.plot(range(1, 121), difference_left, color='green', linewidth=2, label='Left Prediction')

        # Optional: highlight top 3 frames with highest deviation
        top_3_right = np.argsort(difference_right)[-3:] + 1
        top_3_left = np.argsort(difference_left)[-3:] + 1
        for idx in top_3_right:
            plt.axvline(x=idx, color='blue', linestyle='--', alpha=0.3)
        for idx in top_3_left:
            plt.axvline(x=idx, color='green', linestyle='--', alpha=0.3)

        plt.title("Deviation from Professional Pose Across Frames")
        plt.xlabel("Frame")
        plt.ylabel("Euclidean Distance")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    plot_pose_deviation(org_seq, pred_seq_right, pred_seq_left)
