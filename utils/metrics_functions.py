import os
import numpy as np
import sklearn.metrics as skmetr


def write_metrics(threshold, variable_name, res_prob_list, gt_labels_list, output_path):
    # Threshold predictions
    thresholded_preds = apply_threshold(res_prob_list, threshold)
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


def apply_threshold(threshold, preds):
    return (np.array(preds) >= threshold).astype(int)


def calc_percentile(preds, percentile=90):
    # Percentile threshold at 90th percentile
    return np.percentile(preds, percentile)


def calc_static_tresh(preds, static=0.85):
    # Static threshold at 85%
    return static * np.argmax(preds)


def calc_dst_thresh(labels, preds):
    # Uses minimal distance to (0,1) on ROC plot to find best threshold
    fpr, tpr, thresholds = skmetr.roc_curve(np.array(labels), np.array(preds), pos_label=1)
    # Replace NaN values in TPR with 0
    fpr = np.nan_to_num(fpr, nan=0.0)
    tpr = np.nan_to_num(tpr, nan=1.0)
    distances = np.sqrt((fpr) ** 2 + (tpr - 1) ** 2)
    sorted_indices = np.argsort(distances)
    index = sorted_indices[0]  # Second highest index
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
    sorted_indices = np.argsort(distances)
    index = sorted_indices[-1]  # Second highest index (exclude 2.0)
    if index == 0:
        index += 1
    return thresholds[index], fpr[index], tpr[index]


def calc_moving_avg(preds, window_size=20):
    if not len(preds) or window_size <= 0:
        return np.array([])  # Return an empty array for invalid input
    result = []
    for i in range(len(preds)):
        # Compute the average of the last `window_size` elements, or fewer if at the start
        start_index = max(0, i - window_size + 1)
        current_window = preds[start_index:i + 1]
        result.append(np.mean(current_window))
    return np.array(result)


def cumulative_percentile(data, percentile=90):
    # Compute a dynamic moving percentile over a sequence of data.
    result = np.zeros(len(data))
    accumulated_data = []

    for i in range(len(data)):
        accumulated_data.append(data[i])
        result[i] = np.percentile(accumulated_data, percentile)
    return result


'''
def compute_bollinger_bands(data, window_size, num_std_dev=2):
    """Compute Bollinger Bands for the given data."""
    sma = calc_moving_avg(data, window_size)
    rolling_std = np.array([np.std(data[i-window_size+1:i+1])
                            if i >= window_size - 1 else np.nan
                            for i in range(len(data))])
    upper_band = sma + num_std_dev * rolling_std
    lower_band = sma - num_std_dev * rolling_std
    return sma, upper_band, lower_band
'''


'''
def rolling_percentile(data, window_size, percentile):
    """Compute rolling percentile over a data array.
    Args:
        data (list or np.ndarray): The input data.
        window_size (int): The size of the rolling window.
        percentile (float): The desired percentile (0-100).

    Returns:
        np.ndarray: Rolling percentile values with the same length as `data`.
    """
    data = np.asarray(data)
    result = np.zeros(len(data))

    for i in range(len(data)):
        # Dynamically adjust window size for early indices
        window_start = max(0, i - window_size + 1)
        window = data[window_start:i + 1]
        result[i] = np.percentile(window, percentile)
    return result
'''

'''
def calc_avg_tresh(preds):
    # Threshold at 1.5x the average prediction value
    return np.average(preds) * 1.5
'''
