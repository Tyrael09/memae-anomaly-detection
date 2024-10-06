import pandas as pd


def generate_lists(csv_in, clip_len=16, overlap=4, frame_skip=4):
    df = pd.read_csv(csv_in)
    frame_lists = []
    for _, row in df.iterrows():
        if row['label'] == 0:
            start = int(row['start'] * 60) # convert time to frame index
            end = int(row['end'] * 60)
            print(row['video_id'], start, end)
            length = end - start
            num_clips = (length - overlap) // ((clip_len - overlap) * frame_skip) # accounting for overlap and frame rate
            # print(num_clips) # check if value is correct
            next_index = start
            for _ in range(num_clips + 1):
                sub_list = []
                while len(sub_list) < clip_len and next_index <= end:
                    sub_list.append(next_index)
                    next_index += frame_skip # 4
                remaining_frames = end - next_index + frame_skip  # Check how many frames remain
                if remaining_frames < clip_len and len(sub_list) < 16: # If not enough frames are left, increase overlap to fill the last list
                    while len(sub_list) < clip_len:
                       sub_list.insert(0, sub_list[0] - frame_skip)
                frame_lists.append(sub_list)
                # print(sub_list)
                next_index -= overlap * frame_skip # index + 4 - 20 -> index - 16 -> overlap of 4
    return frame_lists

my_csv = '/local/scratch/hendrik/video_annotations_full.csv'
generate_lists(my_csv)