import os
import pandas as pd
import numpy as np
import sklearn.metrics as skmetr
import matplotlib.pyplot as plt

def my_eval_video(data_path, res_path, eval_csv, frames_per_clip=16, normal=True, is_show=True, fps=60):
    df = pd.read_csv(eval_csv)

    gt_labels_list = []
    res_prob_list = []

    gt_labels_list = df["label"].tolist()
    video_names = df["video_id"].unique()

    for video_id in video_names:
        # Load corresponding result file
        res_file_name = f"{video_id}.npy"
        res_file_path = os.path.join(res_path, res_file_name)

        if not os.path.exists(res_file_path):
            print(f"Warning: Result file {res_file_path} not found.")
            continue

        res_prob = np.load(res_file_path)

        if res_prob.size == 0:
            print(f"Warning: {res_file_name} is empty.")
            continue

        res_prob_list.extend(res_prob)

        if normal:
            # Normalize regularity score
            res_prob_norm = res_prob - min(res_prob)
            res_prob = 1 - res_prob_norm / max(res_prob_norm)

        if is_show:
            plt.figure()
            this_video_df = df.loc[df["video_id"] == video_id]

            # Generate time values for the x-axis
            num_clips = len(res_prob)  # TODO: might need to account for overlap? Or not
            time_values = np.arange(num_clips) * (frames_per_clip / fps) * 3  # magic number

            plt.plot(time_values, this_video_df["label"].tolist(), label="Ground Truth")
            plt.plot(time_values, res_prob, label="Predicted")
            plt.ylim(-0.05, 1.05)
            plt.legend()
            plt.xlabel("Time (seconds)")
            plt.ylabel("Score")
            plt.title(f"Ground Truth vs. Predicted Scores: {video_id}")
            plt.savefig(f"{res_path}/{video_id}.png", dpi=300, bbox_inches="tight")
            plt.close()

    # Calculate ROC and AUC
    fpr, tpr, thresholds = skmetr.roc_curve(np.array(gt_labels_list), np.array(res_prob_list), pos_label=1)
    auc = skmetr.auc(fpr, tpr)
    print(f"AUC: {auc}")

    # Save results
    output_path = res_path
    pd.DataFrame({"gt_labels_list": np.double(gt_labels_list)}).to_csv(
        os.path.join(output_path, "gt_labels_all.csv"), index=False
    )
    pd.DataFrame({"est_labels_list": np.double(res_prob_list)}).to_csv(
        os.path.join(output_path, "est_labels_all.csv"), index=False
    )

    with open(os.path.join(output_path, "acc.txt"), "w") as acc_file:
        acc_file.write(f"{data_path}\nAUC: {auc}\n")

    # Calculate the optimal threshold
    distances = np.sqrt((fpr - 0) ** 2 + (tpr - 1) ** 2)
    optimal_index = np.argmin(distances)

    alt_thresh = 0 # TODO: change or parametrise

    if alt_thresh:
        distances = tpr - fpr # 0.9 * np.argmax(distances)?
        optimal_index = np.argmax(distances)

    optimal_threshold = thresholds[optimal_index]
    print(optimal_threshold)
    optimal_fpr = fpr[optimal_index]
    optimal_tpr = tpr[optimal_index]

    # Threshold predictions
    thresholded_preds = (np.array(res_prob_list) >= optimal_threshold).astype(int)

    # Calculate metrics for thresholded predictions
    accuracy = skmetr.accuracy_score(gt_labels_list, thresholded_preds)
    precision = skmetr.precision_score(gt_labels_list, thresholded_preds)
    recall = skmetr.recall_score(gt_labels_list, thresholded_preds)
    f1 = skmetr.f1_score(gt_labels_list, thresholded_preds)

    print(f"Threshold: {optimal_threshold}")
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    # Save metrics to file
    with open(os.path.join(output_path, "metrics.txt"), "w") as metrics_file:
        metrics_file.write(f"Threshold: {optimal_threshold}\n")
        metrics_file.write(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\n")

    # Optional visualization
    if is_show:
        # Thresholded Predictions vs. Ground Truth
        plt.figure()
        time_values = np.arange(len(thresholded_preds)) * (frames_per_clip / fps) * 3  # magic number

        plt.plot(time_values, gt_labels_list, label="Ground Truth")
        plt.step(time_values, thresholded_preds, label="Thresholded Predictions", where="post")
        plt.ylim(-0.05, 1.05)
        plt.legend(loc="upper left")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Label")
        plt.title("Thresholded Predictions vs. Ground Truth")
        plt.savefig(f"{res_path}_thresholded_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Plot the ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random guessing")
        plt.scatter(
            optimal_fpr, optimal_tpr, color="red", label=f"Optimal Threshold = {optimal_threshold:.2f}", marker="o"
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig(f"{res_path}_roc_curve.png", dpi=300, bbox_inches="tight")
        plt.close()

    return auc
