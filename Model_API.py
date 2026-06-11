# ===================== AUTISM BEHAVIOR + ADOS ANALYSIS API =====================
from flask import Flask, request, jsonify
import cv2
import numpy as np
import tempfile
import os
import logging
import pandas as pd
import joblib
from werkzeug.utils import secure_filename
from rtmlib.tools.solution import PoseTracker, Wholebody
import mediapipe as mp
from functools import lru_cache

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_FRAMES = 120
MIN_VALID_FRAMES = 100
BODY_DEVICE = "cpu"
BODY_BACKEND = "onnxruntime"
OPENPOSE_SKELETON = False

# ---------------- FACE MODEL ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------- LOAD ML MODELS ----------------
@lru_cache(maxsize=1)
def load_behavior_models():
    return {
        "scaler_stage1": joblib.load("scaler ST1.joblib"),
        "pca_stage1": joblib.load("PCA ST1.joblib"),
        "model_stage1": joblib.load("XGBoost ST1.joblib"),
        "scaler_stage2": joblib.load("scaler (4).joblib"),
        "pca_stage2": joblib.load("PCA (2).joblib"),
        "model_stage2": joblib.load("XGBoost.joblib")
    }

# ---------------- VIDEO & POSE HELPERS ----------------
def interpolate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 10:
        raise ValueError("Video too short")

    if len(frames) >= TARGET_FRAMES:
        return frames[:TARGET_FRAMES]

    needed = TARGET_FRAMES - len(frames)
    per_gap = needed // (len(frames) - 1)
    out = []

    for i in range(len(frames) - 1):
        out.append(frames[i])
        for j in range(per_gap + 1):
            alpha = (j + 1) / (per_gap + 1)
            out.append(cv2.addWeighted(frames[i], 1 - alpha, frames[i + 1], alpha, 0))
    out.append(frames[-1])
    return out[:TARGET_FRAMES]

def extract_pose_keypoints(frames):
    tracker = PoseTracker(
        Wholebody,
        det_frequency=7,
        tracking=False,
        to_openpose=OPENPOSE_SKELETON,
        mode='performance',
        backend=BODY_BACKEND,
        device=BODY_DEVICE
    )

    indices = [1,2,3,4,5,7,8,9,10,11,13,15,16,17,18,19,20,21,22,23,
               92,95,96,98,100,102,104,106,108,110,112,113,116,117,
               119,121,123,125,127,129,131,133]

    all_points = []
    valid = 0

    for frame in frames:
        keypoints, _ = tracker(frame)
        if keypoints.shape[0] == 0 or keypoints.shape[1] < max(indices):
            continue
        valid += 1
        for p in indices:
            x, y = keypoints[0][p - 1]
            all_points.extend([float(x), float(y)])

    if valid < MIN_VALID_FRAMES:
        raise ValueError("Not enough valid pose frames")
    if len(all_points) != 10080:
        raise ValueError("Invalid feature length")

    return np.array(all_points)

# ---------------- BEHAVIORAL PREDICTION ----------------
def behavior_from_keypoints(points):
    models = load_behavior_models()
    df = pd.DataFrame(points.reshape(1, -1), columns=[f"F{i}" for i in range(1, 10081)])

    s1 = models['scaler_stage1'].transform(df)
    p1 = models['pca_stage1'].transform(s1)
    pred1 = int(models['model_stage1'].predict(p1)[0])
    label1 = {0: 'autistic', 1: 'normal'}.get(pred1)

    label2 = None
    if label1 == 'autistic':
        s2 = models['scaler_stage2'].transform(df)
        p2 = models['pca_stage2'].transform(s2)
        pred2 = int(models['model_stage2'].predict(p2)[0])
        label2 = {0:'armflapping',1:'headbanging',2:'spinning',3:'toe_walking',4:'pacing'}.get(pred2)

    return label1, label2

# ---------------- REPETITIVE & FACIAL ANALYSIS ----------------
def repetitive_score(points):
    pts = points.reshape(-1, 84)
    movement = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    movement = (movement - movement.mean()) / (movement.std() + 1e-6)
    autocorr = np.correlate(movement, movement, mode="full")[movement.size-1:]
    peak = np.max(autocorr[10:40])
    return float(np.clip(peak * 8, 1, 10))

def head_pose_yaw(landmarks, w, h):
    image_points = np.array([
        (landmarks[1].x*w, landmarks[1].y*h),
        (landmarks[152].x*w, landmarks[152].y*h),
        (landmarks[33].x*w, landmarks[33].y*h),
        (landmarks[263].x*w, landmarks[263].y*h),
        (landmarks[61].x*w, landmarks[61].y*h),
        (landmarks[291].x*w, landmarks[291].y*h)
    ], dtype="double")
    model_points = np.array([
        (0.0,0.0,0.0),
        (0.0,-63.6,-12.5),
        (-43.3,32.7,-26),
        (43.3,32.7,-26),
        (-28.9,-28.9,-24.1),
        (28.9,-28.9,-24.1)
    ])
    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([[focal_length,0,center[0]],[0,focal_length,center[1]],[0,0,1]])
    dist = np.zeros((4,1))
    success, rot, trans = cv2.solvePnP(model_points, image_points, camera_matrix, dist)
    rot_mat,_ = cv2.Rodrigues(rot)
    angles,_,_,_,_,_ = cv2.RQDecomp3x3(rot_mat)
    return abs(angles[1])

def facial_analysis(video_path):
    cap = cv2.VideoCapture(video_path)
    total = 0
    looking = 0
    mouth_vals, brow_vals, eye_vals = [], [], []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total += 1
        h,w,_ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            yaw = head_pose_yaw(lm,w,h)
            if yaw < 15:
                looking += 1
            mouth_vals.append(abs(lm[13].y - lm[14].y))
            brow_vals.append(abs(lm[70].y - lm[159].y))
            eye_vals.append(abs(lm[159].y - lm[145].y))
    cap.release()
    eye_score = round(looking/total)*10 if total else 0
    emotion_score = round(np.clip(np.std(mouth_vals)+np.std(brow_vals)+np.std(eye_vals)*200,1,10)) if mouth_vals else 5
    return float(eye_score), float(emotion_score)


# ---------------- TEMP VIDEO HELPER ----------------
def save_temp_video(file):
    temp = tempfile.mkdtemp()
    path = os.path.join(temp, secure_filename(file.filename))
    file.save(path)
    return temp, path

# ---------------- FULL ANALYSIS API ----------------
@app.route("/full-analysis", methods=["POST"])
def full_analysis():
    if "video" not in request.files:
        return jsonify({"error":"No video"}),400
    temp,path = save_temp_video(request.files["video"])
    try:
        frames = interpolate_frames(path)
        points = extract_pose_keypoints(frames)
        stage1, stage2 = behavior_from_keypoints(points)

        if stage1 == "normal":
            return jsonify({
                "stage1_prediction": stage1,
                "combined_score": 0,
                "severity": "None",
                "status": "success"
            })

        rep = repetitive_score(points)
        eye, emotion = facial_analysis(path)
        combined_score = round(rep*0.4 + eye*0.3 + emotion*0.3)
        severity = "Mild" if combined_score<=4 else "Moderate" if combined_score<=7 else "Severe"

        return jsonify({
            "stage1_prediction": stage1,
            "stage2_behavior": stage2,
            "movement_analysis": {"repetitive_score": rep},
            "face_analysis": {"eye_score": eye, "expressiveness_score": emotion},
            "combined_score": combined_score,
            "severity": severity,
            "status": "success"
        })
    finally:
        os.remove(path)
        os.rmdir(temp)

# ---------------- BEHAVIORAL-ONLY API ----------------
@app.route("/behavioral-analysis", methods=["POST"])
def behavioral_analysis_api():
    if "video" not in request.files:
        return jsonify({"error":"No video"}),400
    temp, path = save_temp_video(request.files["video"])
    try:
        frames = interpolate_frames(path)
        points = extract_pose_keypoints(frames)
        stage1, stage2 = behavior_from_keypoints(points)
        return jsonify({
            "stage1_prediction": stage1,
            "stage2_behavior": stage2,
            "status":"success"
        })
    finally:
        os.remove(path)
        os.rmdir(temp)

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)