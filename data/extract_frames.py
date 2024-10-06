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

def extract_all(csv, videos, output, first, last, in_list=None, debugging=False):
    df = pd.read_csv(csv)
    df = df[df['video_id'].apply(lambda x: (int(x.split('_')[1]) >= first and int(x.split('_')[1]) <= last))] # only include >= case_2200
    oob = []
    for _, row in df.iterrows():
        if row['label'] == 0:
            video_name = os.path.join(videos, str(row['video_id']) + ".mp4") # full video path
            #print(video_name)
            start_frame = int(row['start'] * 60)
            end_frame = int(row['end'] * 60)
            if debugging:
                video_capture = cv2.VideoCapture(video_name)
                # Check if video opened successfully
                if not video_capture.isOpened():
                    print(f"Failed to open video: {video_name}")
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
                    output_dir = os.path.join(output, video_id)
                    # print(row['video_id'], in_list)
                    if in_list == None or row['video_id'] in in_list: # if a list was given, compare values. Otherwise, just go ahead
                        print(video_id, start_frame, end_frame)
                        extract_frames(video_name, output_dir, start_frame, end_frame)
    return oob

if __name__ == "__main__":
    # input_videos = ["/local/scratch/Cataract-1K-Full-Videos/case_2000.mp4", "/local/scratch/Cataract-1K-Full-Videos/case_2001.mp4", "/local/scratch/Cataract-1K-Full-Videos/case_2002.mp4"]  
    csv_file = "/local/scratch/hendrik/video_annotations_full.csv"
    video_dir = "/local/scratch/Cataract-1K-Full-Videos/"
    output_frames_dir = '/local/scratch/hendrik/cataract_frames_downsized/' # change path back

    my_list = ["case_2001", "case_2002"] # these were the last videos still missing frames
    missing_videos = extract_all(csv_file, video_dir, output_frames_dir, 0, 3000, in_list=my_list, debugging=False)
    print(missing_videos)
    