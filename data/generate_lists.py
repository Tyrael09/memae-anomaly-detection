import pandas as pd


def generate_lists(
    csv_in, clip_len=16, overlap=4, split=[0, 1], frame_skip=4
):  # don't change frame_skip, would require extracting new frames for all videos
    df = pd.read_csv(csv_in)
    frame_data = []
    for _, row in df.iterrows():
        if row["label"] in split:
            start = int(row["start"] * 60)  # convert time to frame index
            end = int(row["end"] * 60)
            video_id = row["video_id"]
            # print(video_id, start, end)
            length = end - start
            num_clips = (length - overlap) // (
                (clip_len - overlap) * frame_skip
            )  # accounting for overlap and frame rate
            # print(num_clips) # check if value is correct
            next_index = start
            for _ in range(int(num_clips) + 1):
                sub_list = []
                while len(sub_list) < clip_len and next_index <= end:
                    sub_list.append(next_index)
                    next_index += frame_skip  # 4
                remaining_frames = end - next_index + frame_skip  # Check how many frames remain
                if (
                    remaining_frames < clip_len and len(sub_list) < clip_len
                ):  # If not enough frames are left, increase overlap to fill the last list
                    if num_clips > 1:
                        while len(sub_list) < clip_len:
                            sub_list.insert(0, sub_list[0] - frame_skip)
                    else:
                        while len(sub_list) < clip_len:
                            print(f"Insufficient frames for one clip in video {video_id}")
                            sub_list.append(
                                sub_list[-1]
                            )  # bit of a dumb solution, repeating the last frame... but this should rarely happen anyway TODO: think of a better solution for this
                frame_data.append([video_id, sub_list])
                next_index -= overlap * frame_skip  # index + 4 - 20 -> index - 16 -> overlap of 4
    frame_df = pd.DataFrame(frame_data, columns=["video_id", "frame_indices"])
    frame_df.to_csv("/local/scratch/hendrik/test_frame_lists2.csv", index=False)
    return frame_df


my_csv = "/local/scratch/hendrik/test_set.csv"
generate_lists(my_csv)
