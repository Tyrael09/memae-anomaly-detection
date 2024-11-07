import os
import pandas as pd
import numpy as np
import sklearn.metrics as skmetr
import matplotlib.pyplot as plt


def my_eval_video(data_path, res_path, eval_csv, is_show=True):
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
        # res_prob.average()

        if res_prob.size == 0:
            print(f"Warning: {res_file_name} is empty.")
            continue

        res_prob_list.extend(res_prob)
        # normalize regularity score
        res_prob_norm = res_prob - min(res_prob)
        res_prob = 1 - res_prob_norm / max(res_prob_norm)

        if is_show:
            plt.figure()
            this_video_df = df.loc[df["video_id"] == video_id]
            plt.plot(this_video_df["label"].tolist(), label="Ground Truth")
            plt.plot(res_prob, label="Predicted")
            # Set y-axis limits
            plt.ylim(-0.05, 1.05)

            plt.legend()
            # Set custom x-axis labels with video names at regular intervals
            # x_ticks = np.arange(0, len(video_labels), 500)  # Set a larger interval for clarity
            # x_labels = [video_labels[i] for i in x_ticks]
            # plt.xticks(ticks=x_ticks, labels=x_labels, rotation=45, ha='right')
            plt.xlabel("Clips")  # use this for video names later..
            plt.ylabel("Score")
            plt.title(f"Ground Truth vs. Predicted Scores: {video_id}")
            plt.savefig(f"{res_path}/{video_id}.png", dpi=300, bbox_inches="tight")
            plt.close()
            # plt.show(block=True)

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

    # print(f'labels: {len(video_labels)}, predictions: {len(res_prob_list)}')
    # Ensure the length matches
    # assert len(video_labels) == len(res_prob_list), "Mismatch between labels and prediction scores."

    # Optional visualization
    if is_show:
        plt.figure()
        plt.plot(gt_labels_list, label="Ground Truth")
        plt.plot(res_prob_list, label="Predicted")
        plt.legend()
        # Set custom x-axis labels with video names at regular intervals
        # x_ticks = np.arange(0, len(video_labels), 500)  # Set a larger interval for clarity
        # x_labels = [video_labels[i] for i in x_ticks]
        # plt.xticks(ticks=x_ticks, labels=x_labels, rotation=45, ha='right')
        plt.xlabel("Clips")  # use this for video names later..
        plt.ylabel("Score")
        # plt.title('Ground Truth vs. Predicted Scores')
        plt.savefig(f"{res_path}_auc.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Calculate the optimal threshold
    distances = np.sqrt((fpr - 0) ** 2 + (tpr - 1) ** 2)
    optimal_index = np.argmax(distances)
    optimal_threshold = thresholds[optimal_index]
    optimal_fpr = fpr[optimal_index]
    optimal_tpr = tpr[optimal_index]

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random guessing")
    plt.scatter(optimal_fpr, optimal_tpr, color="red", label=f"Optimal Threshold = {optimal_threshold:.2f}", marker="o")

    # Labels and title
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"{res_path}_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    return auc
