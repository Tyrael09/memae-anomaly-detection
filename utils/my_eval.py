import os
import pandas as pd
import numpy as np
import sklearn.metrics as skmetr
import matplotlib.pyplot as plt

def my_eval_video(data_path, res_path, csv_file, is_show=False):
    # Load ground truth labels and results from CSV
    df = pd.read_csv(csv_file)  # Assuming the CSV contains columns for ground truth and predictions

    gt_labels_list = []
    res_prob_list = []
    res_prob_list_org = []

    video_names = df['video_id']
    video_num = len(video_names)

    for vid_ite, video_name in enumerate(video_names):
        print('Eval: %d/%d - %s' % (vid_ite + 1, video_num, video_name))

        # Filter the dataframe by the current video
        video_df = df[df['video_id'] == video_name]

        gt_labels = video_df['label'].values  # ground truth labels
        print(f'gt_labels: {gt_labels}. Are these multiple values?')
        # res file name
        res_file_name = video_name + '.npy'
        res_file_path = os.path.join(res_path, res_file_name)
        res_prob = np.load(res_file_path)  # estimated probability scores

        res_prob_list_org.extend(list(res_prob))
        gt_labels_res = gt_labels[8:-7]  # what's the trimming for??

        # Normalize the predicted scores (regularity score)
        res_prob_norm = res_prob - res_prob.min()
        res_prob_norm = 1 - res_prob_norm / res_prob_norm.max()

        gt_labels_list.extend(list(1 - gt_labels_res + 1))
        res_prob_list.extend(list(res_prob_norm))

    # Calculate ROC and AUC
    fpr, tpr, thresholds = skmetr.roc_curve(np.array(gt_labels_list), np.array(res_prob_list), pos_label=2)
    auc = skmetr.auc(fpr, tpr)
    print(f'AUC: {auc}')

    # Save evaluation metrics (you can replace this with saving to CSV if desired)
    output_path = res_path
    gt_output = {'gt_labels_list': np.double(gt_labels_res)}
    est_output = {'est_labels_list': np.double(res_prob_list)}

    pd.DataFrame(gt_output).to_csv(os.path.join(output_path, video_name + '_gt_label.csv'), index=False)
    pd.DataFrame(est_output).to_csv(os.path.join(output_path, video_name + '_est_label.csv'), index=False)

    # Write AUC result
    with open(os.path.join(output_path, 'acc.txt'), 'w') as acc_file:
        acc_file.write(f'{data_path}\nAUC: {auc}\n')

    # Optional visualization
    if is_show:
        plt.figure()
        plt.plot(gt_labels_list, label='Ground Truth')
        plt.plot(res_prob_list, label='Predicted')
        plt.legend()
        plt.show()

    return auc
