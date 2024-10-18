import cv2
import os
import pandas as pd

'''
Takes a csv with video list, start and end times. Downsizes and saves every "frame_skip"th frame to disk, giving each video its own folder of frames.
'''

def extract_frames(video_path, output_dir, start, end, frame_skip=4):
    os.makedirs(output_dir, exist_ok=True)
    video_capture = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video_capture.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    
    # Ensure start and end are within the valid range
    if start >= total_frames or end >= total_frames:
        print(f"Start or end frame is out of bounds: start={start}, end={end}, total={total_frames}")
    
    if start > end:
        print(f"check the annotations, start={start}, end={end}")

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start) # Set the video capture to start reading from the "start" frame
    success, frame = video_capture.read() # success is a boolean if a frame has been grabbed
    print(f"Initial frame read success: {success}")  # Print success for debugging

    count = start
    while success and count <= end:
        #print(f"Processing frame {count}")  # Debugging print
        frame_filename = os.path.join(output_dir, f"{count}.jpg")
        frame = cv2.resize(frame, (128, 128))  # downsizing the frame
        cv2.imwrite(frame_filename, frame)
        print(f"Saved frame {count} to {frame_filename}")  # Debugging print
        count += frame_skip  # Increment by frame_skip directly
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, count)  # Move the position forward by frame_skip
        success, frame = video_capture.read()  # load the next frame
        if not success:
            print(f"Failed to read frame at position {count}. Exiting loop.")
            break
    video_capture.release()

def extract_all(csv, videos, output, label=0, first=None, last=None, in_list=None, debugging=False): # first, last only used to filter csv during debugging. Not needed if using entire dir anyway.
    df = pd.read_csv(csv)
    if first is not None and last is not None:
        df = df[df['video_id'].apply(lambda x: (int(x.split('_')[1]) >= first and int(x.split('_')[1]) <= last))] # only include >= case_2200
    oob = []
    for _, row in df.iterrows():
        if row['label'] == label:
            video_name = os.path.join(videos, str(row['video_id']) + ".mp4") # full video path
            #print(video_name)
            start_frame = int(row['start'] * 60)
            end_frame = int(row['end'] * 60)
            if debugging:
                video_capture = cv2.VideoCapture(video_name)
                # Check if video opened successfully
                if not video_capture.isOpened():
                    print(f"Failed to open video: {video_name} and debugging: {debugging}")
                    oob.append(video_name)
                total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                # print(f"Total frames in video: {total_frames}")
                # Ensure start and end are within the valid range
                if start_frame >= total_frames or end_frame >= total_frames:
                    print(f"Start or end frame is out of bounds: start={start_frame}, end={end_frame}, total={total_frames}")
                    oob.append([video_name, start_frame, end_frame, total_frames])
            else:
                # if start_frame % 4 != 0: 
                    video_id = os.path.splitext(os.path.basename(video_name))[0]  # Extracts just the video filename without extension
                    # print(video_id, start_frame, end_frame)
                    # print(row['video_id'])
                    output_dir = os.path.join(output, video_id)
                    # print(row['video_id'], in_list)
                    if in_list is None or row['video_id'] in in_list: # if a list was given, compare values. Otherwise, just go ahead
                        print(video_id, start_frame, end_frame)
                        extract_frames(video_name, output_dir, start_frame, end_frame)
    return oob

if __name__ == "__main__": 
    csv_file = "/local/scratch/hendrik/phase_recognition_annotations.csv" # find matching annotations for test set!
    # video_dir = "/local/scratch/Cataract-1K/Lens_irregularity/" # 8 irregular videos + 50 normal test set
    test_set = "/local/scratch/Cataract-1K/Phase_recognition_dataset/videos/"
    output_frames_dir = "/local/scratch/hendrik/cataract_test_frames_downsized/" # for test set

    # missing_videos = extract_all(csv_file, video_dir, output_frames_dir, label=1, debugging=False)
    # print(missing_videos)
    
    extract_all(csv=csv_file, videos=test_set, output=output_frames_dir, label=0)

    # lens_irregularity contains 2595, 3459, 3884, 5419, 5693, 7380, 7764, 7868, 8039, 8041.
    # TODO: find other regular videos
