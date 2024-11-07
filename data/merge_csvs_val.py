import pandas as pd


def assign_labels_to_frames(frame_csv_path, label_csv_path, output_csv_path, fps=60):
    """
    Assigns labels to frame lists in the given frame CSV based on temporal annotations in the label CSV.

    Parameters:
        frame_csv_path (str): Path to the frame list CSV file containing 'video_id' and 'frame_indices'.
        label_csv_path (str): Path to the label CSV file containing 'video_id', 'start', 'end', and 'label'.
        output_csv_path (str): Path to save the merged CSV with assigned labels.
        fps (int): Frame rate to convert time ranges to frame indices (default is 60).

    Returns:
        None: The output is saved to the specified output_csv_path.
    """

    # Load the CSV files
    df_frames = pd.read_csv(frame_csv_path)
    df_labels = pd.read_csv(label_csv_path)

    # Function to assign label to each frame list
    def get_label_for_frame_indices(video_id, frame_indices):
        # Filter the label dataframe for the given video_id
        video_labels = df_labels[df_labels["video_id"] == video_id]

        # Convert start and end times from seconds to frame indices
        video_labels["start_frame"] = video_labels["start"] * fps
        video_labels["end_frame"] = video_labels["end"] * fps

        # Determine the label for this frame list
        frame_labels = []
        for _, row in video_labels.iterrows():
            # Check if all frames in the frame_indices fall within this label range
            if all(row["start_frame"] <= frame <= row["end_frame"] for frame in frame_indices):
                frame_labels.append(row["label"])

        # If there are labels in frame_labels, assign the highest (max) label
        if frame_labels:
            return max(frame_labels)

        # If some frames fall into multiple ranges, assign the max label found in any matching range
        for _, row in video_labels.iterrows():
            if any(row["start_frame"] <= frame <= row["end_frame"] for frame in frame_indices):
                frame_labels.append(row["label"])

        # Return the max label in the case of overlap or no complete match
        return max(frame_labels) if frame_labels else None  # None if no label range matched

    # Apply the function to each row in the frame DataFrame
    df_frames["label"] = df_frames.apply(
        lambda row: get_label_for_frame_indices(row["video_id"], eval(row["frame_indices"])),
        axis=1,
    )

    # Save the updated dataframe to a new CSV
    df_frames.to_csv(output_csv_path, index=False)
    print(f"Merged CSV with labels has been saved as '{output_csv_path}'")


# Example usage:
# assign_labels_to_frames(
#     frame_csv_path='/local/scratch/hendrik/test_frame_lists.csv',
#     label_csv_path='/local/scratch/hendrik/val_set.csv',
#     output_csv_path='/local/scratch/hendrik/merged_csv.csv',
#     fps=60
# )
