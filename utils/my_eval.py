import os
import pandas as pd
import numpy as np
import sklearn.metrics as skmetr
import matplotlib.pyplot as plt

def my_eval_video(data_path, res_path, csv_file, is_show=True):
    # Load ground truth labels and results from CSV
    df = pd.read_csv(csv_file)
    
    # Drop rows where label == 2
    df = df[df['label'] != 2]

    gt_labels_list = []
    res_prob_list = []

    video_names = df['video_id'].unique()
    video_num = len(video_names)

    # Generate the video labels list with video names repeated for each clip
    video_labels = []

    for vid_ite, video_name in enumerate(video_names):
        print(f'Eval: {vid_ite + 1}/{video_num} - {video_name}')

        # Get the ground truth label for the video
        video_df = df[df['video_id'] == video_name]
        video_labels.extend([video_name] * len(video_df)) # only for plotting purposes
        gt_label = video_df['label'].values[0]  # TODO: change this once I have multiple GT values per video!

        # Load corresponding result file
        res_file_name = f"{video_name}.npy"
        res_file_path = os.path.join(res_path, res_file_name)
        
        if not os.path.exists(res_file_path):
            print(f"Warning: Result file {res_file_path} not found.")
            continue

        res_prob = np.load(res_file_path)
        
        if res_prob.size == 0:
            print(f"Warning: {res_file_name} is empty.")
            continue
        
        # Append to lists
        # gt_labels_list.append(gt_label)  # TODO: use this again, once I have annotated the anomalous videos!# Converting to binary for ROC (assuming 1=anomaly, 0=normal)
        
        # Normalise scores
        res_prob_norm = res_prob - res_prob.min() # Shift min score to 0
        res_prob_norm = 1 - res_prob_norm / res_prob_norm.max() # Scale to [0, 1] and invert
        res_prob_list.extend(res_prob) # don't use res_prob_norm
        # Extend gt_labels_list with the same gt_label repeated to match res_prob's length
        gt_labels_list.extend([gt_label] * len(res_prob))

    if not gt_labels_list or not res_prob_list:
        print("Error: No valid data to calculate AUC.")
        return None

    # Calculate ROC and AUC
    fpr, tpr, thresholds = skmetr.roc_curve(np.array(gt_labels_list), np.array(res_prob_list), pos_label=1)
    auc = skmetr.auc(fpr, tpr)
    print(f'AUC: {auc}')

    # Save results
    output_path = res_path
    pd.DataFrame({'gt_labels_list': np.double(gt_labels_list)}).to_csv(
        os.path.join(output_path, 'gt_labels_all.csv'), index=False
    )
    pd.DataFrame({'est_labels_list': np.double(res_prob_list)}).to_csv(
        os.path.join(output_path, 'est_labels_all.csv'), index=False
    )

    with open(os.path.join(output_path, 'acc.txt'), 'w') as acc_file:
        acc_file.write(f'{data_path}\nAUC: {auc}\n')


    #print(f'labels: {len(video_labels)}, predictions: {len(res_prob_list)}')
    # Ensure the length matches
    #assert len(video_labels) == len(res_prob_list), "Mismatch between labels and prediction scores."
    
    # Optional visualization
    if is_show:
        plt.figure()
        plt.plot(gt_labels_list, label='Ground Truth')
        plt.plot(res_prob_list, label='Predicted')
        plt.legend()
        # Set custom x-axis labels with video names at regular intervals
        #x_ticks = np.arange(0, len(video_labels), 500)  # Set a larger interval for clarity
        #x_labels = [video_labels[i] for i in x_ticks]
        #plt.xticks(ticks=x_ticks, labels=x_labels, rotation=45, ha='right')
        plt.xlabel('Clips') # use this for video names later..
        plt.ylabel('Score')
        # plt.title('Ground Truth vs. Predicted Scores')
        plt.savefig(f'{res_path}_auc.png', dpi=300, bbox_inches='tight')
        plt.show(block=True)

    return auc
