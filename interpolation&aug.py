import cv2
import numpy as np
import os


def interpolate_frames(frames, target_frames=120):
    if not frames:
        return frames
    current_frames = len(frames)
    if current_frames == target_frames:
        return frames
    # downsampling
    elif current_frames > target_frames:
        indices = np.linspace(0, current_frames - 1, target_frames, dtype=int)
        return [frames[i] for i in indices]
    # Linear interpolation 
    else:
        interpolated = []
        step = (current_frames - 1) / (target_frames - 1)
        for i in range(target_frames):
            pos = i * step
            lower_idx = int(pos)
            upper_idx = min(lower_idx + 1, current_frames - 1)
            if lower_idx == upper_idx:
                interpolated.append(frames[lower_idx])
            else:
                weight = pos - lower_idx
                frame1 = frames[lower_idx].astype(np.float32)
                frame2 = frames[upper_idx].astype(np.float32)
                interpolated_frame = (1 - weight) * frame1 + weight * frame2
                interpolated.append(interpolated_frame.astype(np.uint8))
        return interpolated


def slow_motion_frames(frames, slow_factor=1.5):
    if not frames:
        return frames
    N = len(frames)
    out = []
    for i in range(N):
        pos = i / slow_factor
        lower_idx = int(pos)
        upper_idx = min(lower_idx + 1, N - 1)
        weight = pos - lower_idx
        f1 = frames[lower_idx].astype(np.float32)
        f2 = frames[upper_idx].astype(np.float32)
        interp = ((1 - weight) * f1 + weight * f2).astype(np.uint8)
        out.append(interp)
    return out

def read_and_interpolate(video_path, target_frames=120):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (640, 480)))
    cap.release()

    frames = interpolate_frames(frames, target_frames)
    fps = 30
    print(f"{os.path.basename(video_path)}: {len(frames)} frames @ {fps} fps "
          f"(duration {len(frames)/fps:.2f}s)")
    return frames, fps

def flip_frames(frames):
    return [cv2.flip(f, 1) for f in frames]

def rotate_frames(frames, angle):
    out = []
    for f in frames:
        h, w = f.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        out.append(rotated)
    return out

def zoom_frames(frames, scale=1.2):
    out = []
    for f in frames:
        h, w = f.shape[:2]
        nw, nh = int(w * scale), int(h * scale)
        enlarged = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x0, y0 = (nw - w)//2, (nh - h)//2
        cropped = enlarged[y0:y0+h, x0:x0+w]
        out.append(cropped)
    return out

def brightness_jitter(frames, factor=1.2):
    out = []
    for f in frames:
        img = np.clip(f.astype(np.float32)*factor, 0, 255).astype(np.uint8)
        out.append(img)
    return out


def save_video(frames, path, fps=30):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"Saved {os.path.basename(path)}: {len(frames)} frames @ {fps} fps "
          f"(duration {len(frames)/fps:.2f}s)")


def preprocess_video(video_path, out_dir="augmented", target_frames=120, slow_factor=None):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]

    frames, fps = read_and_interpolate(video_path, target_frames)

    if slow_factor:
        frames = slow_motion_frames(frames, slow_factor=slow_factor)

    
    save_video(frames, os.path.join(out_dir, f"{base}.mp4"), fps)

    
    save_video(flip_frames(frames), os.path.join(out_dir, f"{base}_flipped.mp4"), fps)
    save_video(rotate_frames(frames, 15), os.path.join(out_dir, f"{base}_rotated_15.mp4"), fps)
    save_video(rotate_frames(frames, -10), os.path.join(out_dir, f"{base}_rotated_minus10.mp4"), fps)
    save_video(zoom_frames(frames, 1.2), os.path.join(out_dir, f"{base}_zoomed.mp4"), fps)
    save_video(brightness_jitter(frames, 1.2), os.path.join(out_dir, f"{base}_jittered.mp4"), fps)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Autism video preprocessing: 120 frames @ 30 FPS")
    parser.add_argument("--input", "-i", required=True, help="Input video file")
    parser.add_argument("--out", "-o", default="augmented", help="Output folder")
    parser.add_argument("--slow", "-s", type=float, default=None,
                        help="Optional slow motion factor (>1 for slower motion)")
    args = parser.parse_args()

    preprocess_video(args.input, args.out, target_frames=120, slow_factor=args.slow)
