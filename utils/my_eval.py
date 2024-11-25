import os
import pandas as pd
import numpy as np
import sklearn.metrics as skmetr
import matplotlib.pyplot as plt

def my_eval_video(res_path, eval_csv, frames_per_clip=16, normal=True, fps=60):
    def write_metrics(threshold, variable_name):
        # Threshold predictions
        thresholded_preds = thresholded_values(res_prob_list, threshold)
        # Calculate metrics for thresholded predictions
        accuracy = skmetr.accuracy_score(gt_labels_list, thresholded_preds)
        precision = skmetr.precision_score(gt_labels_list, thresholded_preds)
        recall = skmetr.recall_score(gt_labels_list, thresholded_preds)
        f1 = skmetr.f1_score(gt_labels_list, thresholded_preds)

        # Save metrics to file
        with open(os.path.join(output_path, f"metrics_roc_{variable_name}.txt"), "w") as metrics_file:
            metrics_file.write(f"Threshold: {threshold}\n")
            metrics_file.write(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\n")

        return thresholded_preds
    

    def thresholded_values(threshold, preds):
        return (np.array(preds) >= threshold).astype(int)


    def calc_pct_thresh(preds, percentile=90):
        # Percentile threshold at 90th percentile
        return np.percentile(preds, percentile)


    def calc_static_tresh(preds, static=0.85):
        # Static threshold at 85%
        return static * np.argmax(preds)


    def calc_avg_tresh(preds):
        # Threshold at 1.5x the average prediction value
        return np.average(preds) * 1.5


    def calc_dst_thresh(labels, preds):
        # Uses minimal distance to (0,1) on ROC plot to find best threshold
        fpr, tpr, thresholds = skmetr.roc_curve(np.array(labels), np.array(preds), pos_label=1)
        # Replace NaN values in TPR with 0
        fpr = np.nan_to_num(fpr, nan=0.0)
        tpr = np.nan_to_num(tpr, nan=1.0)
        distances = np.sqrt((fpr) ** 2 + (tpr - 1) ** 2)
        # index = np.argmin(distances)
        # Find the indices of the two highest values
        sorted_indices = np.argsort(distances)  # Sort distances in ascending order
        index = sorted_indices[0]  # Second highest is the second last index after sorting
        if index == 0:
            index += 1
        return thresholds[index], fpr[index], tpr[index]
    

    def calc_tpr_thresh(labels, preds):
        # Maximise distance to FPR == TPR (random guessing in balanced datasets) for threshold
        fpr, tpr, thresholds = skmetr.roc_curve(np.array(labels), np.array(preds), pos_label=1)
        # Replace NaN values in TPR with 0
        fpr = np.nan_to_num(fpr, nan=0.0)
        tpr = np.nan_to_num(tpr, nan=1.0)
        distances = tpr - fpr 
        #print(distances)
        # index = np.argmax(distances)
        # Find the indices of the two highest values
        sorted_indices = np.argsort(distances)  # Sort distances in ascending order
        index = sorted_indices[-1]  # Second highest is the second last index after sorting
        #print(index)
        #print(thresholds[index])
        #print(thresholds)
        if index == 0:
            index += 1
        return thresholds[index], fpr[index], tpr[index]
    

    def make_plot(threshold, comment, name, time, video_df, preds):
        epsilon = 0.005
        plt.figure()
        plt.plot(time, video_df["label"].tolist(), label="Ground Truth")
        plt.plot(time, preds + epsilon, label="Predicted")
        plt.plot(time, [threshold] * len(time), label=comment, linestyle=":")
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Score")
        plt.title(f"Ground Truth vs. Predicted Scores: {video_id}")
        plt.savefig(f"{res_path}/{video_id}_{name}.png", dpi=300, bbox_inches="tight")
        plt.close()


    df = pd.read_csv(eval_csv)
    gt_labels_list = []
    res_prob_list = []
    gt_labels_list = df["label"].tolist()
    video_names = df["video_id"].unique()

    for video_id in video_names:
        # Load corresponding result file
        res_file_name = f"{video_id}.npy"
        res_file_path = os.path.join(res_path, res_file_name)
        this_video_df = df.loc[df["video_id"] == video_id]

        if not os.path.exists(res_file_path):
            print(f"Warning: Result file {res_file_path} not found.")
            continue

        res_prob = np.load(res_file_path)

        if res_prob.size == 0:
            print(f"Warning: {res_file_name} is empty.")
            continue

        if normal:
            # Normalize regularity score
            res_prob_norm = res_prob - min(res_prob)
            res_prob = 1 - res_prob_norm / max(res_prob_norm)

        res_prob_list.extend(res_prob)

        avg_threshold = calc_avg_tresh(res_prob)
        pct_threshold = calc_pct_thresh(res_prob)
        dst_threshold, _, _ = calc_dst_thresh(this_video_df["label"].tolist(), res_prob)
        tpr_threshold, _, _ = calc_tpr_thresh(this_video_df["label"].tolist(), res_prob)

        # Generate time values for the x-axis
        num_clips = len(res_prob) 
        time_values = np.arange(num_clips) * (frames_per_clip / fps) * 3  # magic number

        make_plot(avg_threshold, "Threshold (1.5x average)", "avg", time_values, this_video_df, res_prob)
        make_plot(pct_threshold, "Threshold (90th Percentile)", "pct", time_values, this_video_df, res_prob)
        make_plot(dst_threshold, "Threshold (ROC distance to (0,1))", "dst", time_values, this_video_df, res_prob)
        make_plot(tpr_threshold, "Threshold (TPR-FPR)", "tpr", time_values, this_video_df, res_prob)

        avg_prob_thres = thresholded_values(avg_threshold, res_prob)
        pct_prob_thres = thresholded_values(pct_threshold, res_prob)
        dst_prob_thres = thresholded_values(dst_threshold, res_prob)
        tpr_prob_thres = thresholded_values(tpr_threshold, res_prob)

        make_plot(avg_threshold, "Threshold (1.5x average)", "avg_thresh", time_values, this_video_df, avg_prob_thres)
        make_plot(pct_threshold, "Threshold (90th Percentile)", "pct_thresh", time_values, this_video_df, pct_prob_thres)
        make_plot(dst_threshold, "Threshold (ROC distance to (0,1))", "dst_thresh", time_values, this_video_df, dst_prob_thres)
        make_plot(tpr_threshold, "Threshold (TPR-FPR)", "tpr_thresh", time_values, this_video_df, tpr_prob_thres)

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
 
    optimal_threshold, optimal_fpr, optimal_tpr = calc_dst_thresh(gt_labels_list, res_prob_list)
    thresholded_preds = write_metrics(optimal_threshold, "distance_threshold")

    tpr_threshold, _, _ = calc_tpr_thresh(gt_labels_list, res_prob_list)
    write_metrics(tpr_threshold, "tpr_threshold")
    
    static_threshold = calc_static_tresh(res_prob_list)
    write_metrics(static_threshold, "static_threshold")
    
    percentile_threshold = calc_pct_thresh(res_prob_list, 90)
    write_metrics(percentile_threshold, "percentile_threshold")
    

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
    plt.savefig(f"{res_path}/thresholded_comparison_all_clips.png", dpi=300, bbox_inches="tight")
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
    plt.savefig(f"{res_path}/roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
