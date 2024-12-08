import os
import pandas as pd
import numpy as np
import sklearn.metrics as skmetr
import matplotlib.pyplot as plt
import pathlib
import metrics_functions as metrics

def my_eval_video(res_path, eval_csv, frames_per_clip=16, normal=True, fps=60, index=10):
    new_char = '8'  # Replace '6' with '7'
    output_path = res_path[:index] + new_char + res_path[index + 1:] # only changed folder index
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 

    def make_plot(thresholds, legends, time, video_df, preds, thresh_types):
        epsilon = 0.005
        plt.plot(time, video_df["label"].tolist(), label="Ground Truth")
        plt.plot(time, preds + epsilon, label="Predicted")
        for threshold, legend, thresh_type in zip(thresholds, legends, thresh_types):
            if thresh_type == 0:
                threshold = [threshold] * len(time)
            plt.plot(time, threshold, label=legend, linestyle="--")
        plt.ylim(-0.05, 1.05)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Score")
        plt.title(f"Ground Truth vs. Predicted Scores: {video_id}")
        plt.legend(fontsize="small")
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large') 
        plt.savefig(f"{output_path}/{video_id}_thresholds_{thresh_type}.png", dpi=300, bbox_inches="tight")
        plt.close()


    df = pd.read_csv(eval_csv)
    gt_labels_list = []
    res_prob_list = []
    gt_labels_list = df["label"].tolist()
    video_names = df["video_id"].unique()
    results = {}

    for video_id in video_names:
        # Load corresponding result file
        res_file_name = f"{video_id}.npy"
        res_file_path = os.path.join(res_path, res_file_name)
        this_video_df = df.loc[df["video_id"] == video_id]

        if not os.path.exists(res_file_path):
            print(f"Warning: Result file {res_file_path} not found.")
            continue

        # Load predictions
        res_prob = np.load(res_file_path)

        if res_prob.size == 0:
            print(f"Warning: {res_file_name} is empty.")
            continue

        # Normalize regularity score
        if normal:
            res_prob_norm = res_prob - min(res_prob)
            res_prob = 1 - res_prob_norm / max(res_prob_norm)

        avg_score = np.mean(res_prob)
        res_prob_list.extend(res_prob)

        # Calculate thresholds
        pct_threshold = metrics.calc_percentile(res_prob)
        tpr_threshold, _, _ = metrics.calc_tpr_thresh(this_video_df["label"].tolist(), res_prob)
        sma = metrics.calc_moving_avg(res_prob, 20)
        cumulative_pct = metrics.cumulative_percentile(res_prob, 90) 

        # Generate time values for the x-axis
        num_clips = len(res_prob) 
        time_values = np.arange(num_clips) * (frames_per_clip / fps) * 3  # magic number

        thresholds = [pct_threshold, tpr_threshold, sma, cumulative_pct]

        threshold_preds = {}  # Store predictions and threshold names
        legends = ["90th Percentile", "TPR-FPR", "SMA","Moving 90th Percentile"]

        for t, name in zip(thresholds, legends):
            threshold_preds[name] = metrics.apply_threshold(t, avg_score)

        # Save results to dictionary
        results[video_id] = {
            "avg_score": avg_score,
            "label": this_video_df["label"].iloc[0],  # TODO: FIX, take the MAX of all labels!
            "binary_pred": threshold_preds["90th Percentile"], # TODO: something also going wrong here..
        }
        
        types = [0, 0, 1, 1, 1]
        make_plot(thresholds[:2], legends[:2], time_values, this_video_df, res_prob, types[:2])
        make_plot(thresholds[2:], legends[2:], time_values, this_video_df, res_prob, types[2:])


    # Calculate ROC and AUC
    fpr, tpr, thresholds = skmetr.roc_curve(np.array(gt_labels_list), np.array(res_prob_list), pos_label=1)
    auc = skmetr.auc(fpr, tpr)
    print(f"AUC: {auc}")

    # Calculate Precision-Recall curve and PR-AUC
    precision, recall, thresholds_pr = skmetr.precision_recall_curve(np.array(gt_labels_list), np.array(res_prob_list), pos_label=1)
    pr_auc = skmetr.auc(recall, precision)
    print(f"PR AUC: {pr_auc}")

    # Save metrics to file
    with open(os.path.join(output_path, f"metrics_auc.txt"), "w") as file:
        file.write(f"AUROC: {auc} \nPR-AUC: {pr_auc}")
    # Save results
    pd.DataFrame({"gt_labels_list": np.double(gt_labels_list)}).to_csv(
        os.path.join(output_path, "gt_labels_all.csv"), index=False
    )
    pd.DataFrame({"est_labels_list": np.double(res_prob_list)}).to_csv(
        os.path.join(output_path, "est_labels_all.csv"), index=False
    )
 
    optimal_threshold, optimal_fpr, optimal_tpr = metrics.calc_dst_thresh(gt_labels_list, res_prob_list)
    thresholded_preds = metrics.write_metrics(optimal_threshold, "distance_threshold", res_prob_list, gt_labels_list, output_path)

    tpr_threshold, _, _ = metrics.calc_tpr_thresh(gt_labels_list, res_prob_list)
    metrics.write_metrics(tpr_threshold, "tpr_threshold", res_prob_list, gt_labels_list, output_path)
    
    static_threshold = metrics.calc_static_tresh(res_prob_list)
    metrics.write_metrics(static_threshold, "static_threshold", res_prob_list, gt_labels_list, output_path)
    
    percentile_threshold = metrics.calc_percentile(res_prob_list, 90)
    metrics.write_metrics(percentile_threshold, "percentile_threshold", res_prob_list, gt_labels_list, output_path)

    cumulative_percentile_threshold = metrics.cumulative_percentile(res_prob_list, 90)
    metrics.write_metrics(cumulative_percentile_threshold, "cumulative_percentile_threshold", res_prob_list, gt_labels_list, output_path)
    

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
    plt.savefig(f"{output_path}/thresholded_comparison_all_clips.png", dpi=300, bbox_inches="tight")
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
    plt.savefig(f"{output_path}/roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


    def plot_video_level_results(video_scores, output_path):
        """
        Generate a plot of video-level predictions vs ground truth labels.

        Args:
            video_scores (dict): Dictionary with video-level predictions and labels.
            output_path (str): Path to save the plot.
        """
        video_ids = list(video_scores.keys())  # Extract video IDs from the dict keys
        avg_preds = [video_scores[vid]["avg_score"] for vid in video_ids]  # Access avg_score
        labels = [video_scores[vid]["label"] for vid in video_ids]  # Access labels

        plt.figure(figsize=(10, 6))
        plt.bar(video_ids, avg_preds, color="blue", alpha=0.6, label="Predicted Scores")
        plt.scatter(video_ids, labels, color="red", label="Ground Truth", zorder=5)
        plt.xticks(rotation=90, fontsize=8)
        plt.ylabel("Score/Label")
        plt.title("Video-Level Predictions vs Ground Truth")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_path}/video_level_predictions.png", dpi=300)
        plt.close()


    # Generate plot
    plot_video_level_results(results, output_path)

    # Calculate metrics
    # Updated metric calculations
    video_labels = [results[vid]["label"] for vid in results]  # Extract labels from the dict
    print(results["case_2013"]["label"]) # TODO: problem is that label is just one value while binary_pred is a list!
    print(results["case_2013"]["binary_pred"]) # this is actually binary.. so what's the problem?
    video_preds = [results[vid]["binary_pred"] for vid in results]  # Extract predictions from the dict
    video_accuracy = skmetr.accuracy_score(video_labels, video_preds)
    video_precision = skmetr.precision_score(video_labels, video_preds)
    video_recall = skmetr.recall_score(video_labels, video_preds)
    video_f1 = skmetr.f1_score(video_labels, video_preds)

    # Save metrics
    with open(os.path.join(output_path, "video_level_metrics.txt"), "w") as file:
        file.write(f"Accuracy: {video_accuracy}\n")
        file.write(f"Precision: {video_precision}\n")
        file.write(f"Recall: {video_recall}\n")
        file.write(f"F1 Score: {video_f1}\n")
