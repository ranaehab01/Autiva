import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import logging
from rtmlib.tools.solution import PoseTracker, Wholebody

# init variables
MAX_FRAMES = 120
KEYPOINTS_PER_FRAMES = 84  # 42 landmarks * (x,y)
DEVICE = 'cpu'
OUTPUT_CSV = "armflapping_keypoints_output.csv"
OPENPOSE_SKELETON = False
BODY_BACKEND = 'onnxruntime'
BODY_DEVICE = 'cpu'

# ==========================================================
# INITIALIZE MODELS
# ==========================================================
# mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
pose_mp = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# awl haga n3rf el model(WholeBody)
def initialize_body_tracker():
    try:
        return PoseTracker(
            Wholebody,
            det_frequency=7,
            tracking=False,
            to_openpose=OPENPOSE_SKELETON,
            mode='performance',
            backend=BODY_BACKEND,
            device=BODY_DEVICE
        )
    except Exception as e:
        logger.error(f"Error initializing body tracker: {e}")
        return None


def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return None
    
    all_points = []
    valid_frames = 0
    frame_count = 0
    expected_keypoints = 42
    keypoint_indices = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                        92, 95, 96, 98, 100, 102, 104, 106, 108, 110, 112, 113, 116, 117,
                        119, 121, 123, 125, 127, 129, 131, 133]

    body_tracker = initialize_body_tracker()

# openCV bytl3 BGR
    while frame_count < MAX_FRAMES:
        ret, frame = cap.read()     
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        
       # nt2kd mn el points w filter el frames
        try:
            keypoints = body_tracker(frame_rgb)
            
            # lw msh shayf el sha5s
            if keypoints.shape[0] == 0:
                logger.debug(f"Frame {frame_count + 1}: No keypoints detected")
                frame_count += 1
                continue
            
            # lw mrg3 keypoints 2a2l mn akbr keypoint 3andy
            if keypoints.shape[1] < max(keypoint_indices):
                logger.debug(f"Frame {frame_count + 1}: Expected at least {max(keypoint_indices)} keypoints, got {keypoints.shape[1]}")
                frame_count += 1
                continue
                
            valid_frames += 1
            frame_points = []
             
            # hntl3 el keypoints 
            for p in keypoint_indices:
                try:
                    # keypoints 0 index wana bad2a mn 1 f hst5dm p-1
                    x, y = keypoints[0][p - 1]
                    frame_points.extend([float(x), float(y)])
                    # 3shan a5ly el length sabt
                except IndexError as e:
                    logger.error(f"Keypoint index error at frame {frame_count + 1}, index {p - 1}: {e}")
                    frame_points.extend([0.0, 0.0])
            
            # validation lel points = 84
            if len(frame_points) != expected_keypoints * 2:
                logger.warning(f"Frame {frame_count + 1}: Expected {expected_keypoints * 2} coordinates, got {len(frame_points)}")
                
                # lw zyada trunc lw 2olayel padding
                if len(frame_points) > expected_keypoints * 2:
                    frame_points = frame_points[:expected_keypoints * 2]
                else:
                    frame_points.extend([0.0] * (expected_keypoints * 2 - len(frame_points)))
            
            all_points.extend(frame_points)
            # lw 7asl error l ay sabb add 84 zeros 3shan el length yfdl sabt
        except Exception as e:
            logger.error(f"Error processing frame {frame_count + 1}: {e}")
            all_points.extend([0.0] * (expected_keypoints * 2))
        
        frame_count += 1

    cap.release()
    
    logger.info(f"Processed {frame_count} total frames, {valid_frames} valid frames with keypoints")
    
    # validation lel frame----> pad with zeros
    if frame_count < MAX_FRAMES:
        missing_frames = MAX_FRAMES - frame_count
        missing_coordinates = missing_frames * expected_keypoints * 2
        all_points.extend([0.0] * missing_coordinates)
        logger.info(f"Padded with {missing_frames} empty frames")
    
    # Final validation
    expected_total_coordinates = MAX_FRAMES * expected_keypoints * 2
    if len(all_points) != expected_total_coordinates:
        logger.warning(f"Expected {expected_total_coordinates} total coordinates, got {len(all_points)}")
        if len(all_points) > expected_total_coordinates:
            all_points = all_points[:expected_total_coordinates]
        else:
            all_points.extend([0.0] * (expected_total_coordinates - len(all_points)))
    
    if valid_frames < 100:
        logger.warning(f"Only {valid_frames} frames had detectable keypoints; need at least 100 for good results.")
    
    logger.info(f"Successfully extracted {len(all_points)} coordinates from {valid_frames} valid frames")
    return np.array(all_points)

# -------------------------------------------------------------------------------------------------------
def find_videos_in_folder(folder_path):
    """Find all video files in the given folder (including subfolders)"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.MP4', '.AVI', '.MOV', '.MKV'}
    video_files = []
    
    if not os.path.exists(folder_path):
        logger.error(f" Folder does not exist: {folder_path}")
        return video_files
    
    logger.info(f"Searching for videos in: {folder_path}")
    
    
    all_items = os.listdir(folder_path)
    logger.info(f"Folder contents: {all_items}")
    
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in video_extensions:
                video_path = os.path.join(root, file)
                video_files.append((video_path, file))
                logger.info(f"Found video: {file}")
    
    logger.info(f"Found {len(video_files)} video files total")
    return video_files


def explore_folder_structure(folder_path):
    print(f"\n EXPLORING FOLDER STRUCTURE: {folder_path}")
    print("=" * 60)
    
    if not os.path.exists(folder_path):
        print(f" Folder does not exist: {folder_path}")
        return
    
    for root, dirs, files in os.walk(folder_path):
        level = root.replace(folder_path, '').count(os.sep)
        indent = '  ' * level
        print(f"{indent} {os.path.basename(root)}/")
        
        sub_indent = '  ' * (level + 1)
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                print(f"{sub_indent} {file}")
            else:
                print(f"{sub_indent} {file}")


def process_video_folder(input_folder):
    

    explore_folder_structure(input_folder)
    
    videos = find_videos_in_folder(input_folder)
    
    if not videos:
        logger.error(" No videos found in the folder!")
        return
    
    all_data = []
    successful_count = 0
    
    print(f"\n Processing {len(videos)} videos from: {input_folder}")
    print("=" * 60)
    
    for i, (video_path, video_name) in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] Processing: {video_name}")
        print(f"    Path: {video_path}")
        
        # Extract keypoints
        keypoints = extract_keypoints(video_path)
        
        if keypoints is not None:
            all_data.append([video_name] + keypoints.tolist())
            successful_count += 1
            print(f"    Successfully processed: {video_name}")
        else:
            print(f"    Failed to process: {video_name}")
    
    # Save all data to CSV
    if all_data:
        file_exists = os.path.isfile(OUTPUT_CSV)
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["video_name"] + [f"kp_{i+1}" for i in range(len(all_data[0]) - 1)]
            writer.writerow(header)
            writer.writerows(all_data)
        
        print(f"\n Saved {successful_count}/{len(videos)} videos to {OUTPUT_CSV}")
        print(f" Success rate: {(successful_count/len(videos))*100:.1f}%")
    else:
        print(" No videos were successfully processed!")


if __name__ == "__main__":
    possible_paths = [
        r"D:\Autism_Detection_GP\Coding2\autism_dataset\aug_arm"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            input_folder = path
            print(f" Using folder: {input_folder}")
            break
    else:
        input_folder = r"D:\Autism_Detection_GP\Coding2\autism_dataset\sub_arm"
        print(f" Using default folder: {input_folder}")
    
    # Process all videos in the folder
    process_video_folder(input_folder)