from __future__ import annotations

import os
import cv2
import subprocess
import time
from collections import deque
from typing import Dict, Any, List, Optional, Callable, Tuple

import supervision as sv
from inference import get_model

try:
    from sports.basketball import ShotEventTracker
except Exception as e:
    raise SystemExit(
        "Failed to import ShotEventTracker from sports.basketball. "
        "Make sure you installed the sports repo from the feat/basketball branch. "
        "See README in this folder."
    ) from e

# --------- Constants (match your model) ---------
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.7
BALL_IN_BASKET_CLASS_ID = 1
JUMP_SHOT_CLASS_ID = 5
LAYUP_DUNK_CLASS_ID = 6

# Replace if yours differs
PLAYER_DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"


class Settings:
    def __init__(
        self,
        api_key: str,
        output_path: str = "outputs/annotated.mp4",
        clips_dir: str = "clips",
        max_dim: int = 1024,
        every_n: int = 1,
        context_sec: float = 3.0,
        save_annotated: bool = True,
    ):
        self.api_key = api_key
        self.output_path = output_path
        self.clips_dir = clips_dir
        self.max_dim = int(max_dim)
        self.every_n = max(1, int(every_n))
        self.context_sec = float(context_sec)
        self.save_annotated = bool(save_annotated)


def read_api_key(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            key = f.read().strip()
            return key or None
    except FileNotFoundError:
        return None


def make_writer(path: str, width: int, height: int, fps: float) -> Tuple[cv2.VideoWriter, str]:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if writer is not None and writer.isOpened():
        return writer, path

    # Fallbacks (AVI)
    alt_path = os.path.splitext(path)[0] + ".avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(alt_path, fourcc, fps, (width, height))
    if writer is not None and writer.isOpened():
        return writer, alt_path

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(alt_path, fourcc, fps, (width, height))
    if writer is None or not writer.isOpened():
        raise RuntimeError("Failed to open any VideoWriter. Try reducing resolution, FPS, or codec.")
    return writer, alt_path


def maybe_compress_with_ffmpeg(path: str) -> Optional[str]:
    if not path.lower().endswith(".mp4"):
        return None
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        return None
    compressed = path.replace(".mp4", "-compressed.mp4")
    cmd = ["ffmpeg", "-y", "-i", path, "-vcodec", "libx264", "-crf", "28", compressed]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return compressed if os.path.exists(compressed) else None
    except subprocess.CalledProcessError:
        return None


def ffmpeg_concat_mp4s(input_files: List[str], output_path: str) -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        return False
    list_path = os.path.splitext(output_path)[0] + "_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in input_files:
            p_escaped = p.replace("\\", "\\\\")
            f.write(f"file '{p_escaped}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", output_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass


def opencv_concat_videos(input_files: List[str], output_path: str, fps: float, size: Tuple[int, int]) -> str:
    writer, out_path = make_writer(output_path, size[0], size[1], fps)
    for p in input_files:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            continue
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
        cap.release()
    writer.release()
    return out_path


def process_video(
    video_path: str,
    settings: Settings,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Dict[str, Any]:
    """
    progress_callback(done_frames, total_frames, elapsed_seconds)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    os.makedirs(settings.clips_dir, exist_ok=True)

    # Load model (Roboflow Inference CPU)
    model = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=settings.api_key)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open input video")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resize plan
    scale_w, scale_h = W, H
    if settings.max_dim and max(W, H) > settings.max_dim:
        if W >= H:
            scale_w = settings.max_dim
            scale_h = int(H * (settings.max_dim / W))
        else:
            scale_h = settings.max_dim
            scale_w = int(W * (settings.max_dim / H))

    writer, written_path = make_writer(settings.output_path, scale_w, scale_h, fps)

    # Annotations
    COLOR = sv.ColorPalette.from_hex(
        [
            "#ffff00",
            "#ff9b00",
            "#ff66ff",
            "#3399ff",
            "#ff66b2",
            "#ff8080",
            "#b266ff",
            "#9999ff",
            "#66ffff",
            "#33ff99",
            "#66ff66",
            "#99ff00",
        ]
    )
    box_annotator = sv.BoxAnnotator(color=COLOR, thickness=2)
    label_annotator = sv.LabelAnnotator(color=COLOR, text_color=sv.Color.BLACK)

    reset_time_frames = int(fps * 1.7)
    minimum_frames_between_starts = int(fps * 0.5)
    cooldown_frames_after_made = int(fps * 1.0)

    shot_event_tracker = ShotEventTracker(
        reset_time_frames=reset_time_frames,
        minimum_frames_between_starts=minimum_frames_between_starts,
        cooldown_frames_after_made=cooldown_frames_after_made,
    )

    made_total = 0
    missed_total = 0

    pre_frames = int(round(settings.context_sec * fps))
    post_frames = int(round(settings.context_sec * fps))
    pre_buffer_annot = deque(maxlen=pre_frames)
    pre_buffer_raw = deque(maxlen=pre_frames)

    current_clip = None
    made_clip_paths: List[str] = []
    miss_clip_paths: List[str] = []
    made_idx = 0
    miss_idx = 0

    def start_clip(event_type: str, first_frame_for_clip):
        nonlocal made_idx, miss_idx, current_clip
        if event_type == "MADE":
            made_idx += 1
            basename = f"made_{made_idx:03d}.mp4"
        else:
            miss_idx += 1
            basename = f"miss_{miss_idx:03d}.mp4"
        clip_path = os.path.join(settings.clips_dir, basename)
        clip_writer, clip_path_real = make_writer(clip_path, scale_w, scale_h, fps)

        if settings.save_annotated:
            for f in pre_buffer_annot:
                clip_writer.write(f)
        else:
            for f in pre_buffer_raw:
                if (f.shape[1], f.shape[0]) != (scale_w, scale_h):
                    f = cv2.resize(f, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
                clip_writer.write(f)

        clip_writer.write(first_frame_for_clip)
        current_clip = {
            "type": event_type,
            "writer": clip_writer,
            "path": clip_path_real,
            "frames_left": post_frames,
        }

    def finish_clip():
        nonlocal current_clip
        if current_clip is None:
            return
        current_clip["writer"].release()
        if current_clip["type"] == "MADE":
            made_clip_paths.append(current_clip["path"])
        else:
            miss_clip_paths.append(current_clip["path"])
        current_clip = None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = 0
    write_index = 0

    start_time = time.time()
    # throttle progress callbacks to ~10fps of reports
    last_reported_chunk = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # skip frames for speed
        if settings.every_n > 1 and (frame_index % settings.every_n != 0):
            frame_index += 1
            # still keep the progress moving
            if progress_callback and total_frames > 0:
                cur_chunk = frame_index // 10
                if cur_chunk != last_reported_chunk:
                    last_reported_chunk = cur_chunk
                    progress_callback(min(frame_index, total_frames), total_frames, time.time() - start_time)
            continue

        # resize if needed
        frame_proc = (
            cv2.resize(frame, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
            if (scale_w, scale_h) != (W, H)
            else frame
        )

        # inference
        result = model.infer(frame_proc, confidence=CONFIDENCE_THRESHOLD, iou_threshold=IOU_THRESHOLD)[0]
        detections = sv.Detections.from_inference(result)

        has_jump_shot = len(detections[detections.class_id == JUMP_SHOT_CLASS_ID]) > 0
        has_layup_dunk = len(detections[detections.class_id == LAYUP_DUNK_CLASS_ID]) > 0
        has_ball_in_basket = len(detections[detections.class_id == BALL_IN_BASKET_CLASS_ID]) > 0

        labels: List[str] = []
        for cid in detections.class_id.tolist():
            if cid == JUMP_SHOT_CLASS_ID:
                labels.append("jump_shot")
            elif cid == LAYUP_DUNK_CLASS_ID:
                labels.append("layup/dunk")
            elif cid == BALL_IN_BASKET_CLASS_ID:
                labels.append("ball_in_basket")
            else:
                labels.append(str(cid))

        annotated = box_annotator.annotate(scene=frame_proc.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        events = shot_event_tracker.update(
            frame_index=write_index,
            has_jump_shot=has_jump_shot,
            has_layup_dunk=has_layup_dunk,
            has_ball_in_basket=has_ball_in_basket,
        )

        if events:
            for e in events:
                if e["event"] == "MADE":
                    made_total += 1
                elif e["event"] == "MISSED":
                    missed_total += 1

        # HUD
        cv2.rectangle(annotated, (10, 10), (320, 110), (0, 0, 0), -1)
        cv2.putText(annotated, f"Made:   {made_total}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(annotated, f"Missed: {missed_total}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        writer.write(annotated)

        # buffers
        pre_buffer_annot.append(annotated.copy())
        pre_buffer_raw.append(frame_proc.copy())

        # event/clip logic
        if events:
            event_labels = [e["event"] for e in events]
            if "MADE" in event_labels:
                event_type = "MADE"
            elif "MISSED" in event_labels:
                event_type = "MISSED"
            else:
                event_type = None

            if event_type is not None:
                frame_for_clip = annotated if settings.save_annotated else frame_proc
                if current_clip is None:
                    start_clip(event_type, frame_for_clip)
                else:
                    current_clip["frames_left"] = max(current_clip["frames_left"], post_frames)

        if current_clip is not None:
            frame_for_clip = annotated if settings.save_annotated else frame_proc
            current_clip["writer"].write(frame_for_clip)
            current_clip["frames_left"] -= 1
            if current_clip["frames_left"] <= 0:
                finish_clip()

        frame_index += 1
        write_index += 1

        if progress_callback and total_frames > 0:
            cur_chunk = frame_index // 10
            if cur_chunk != last_reported_chunk:
                last_reported_chunk = cur_chunk
                progress_callback(min(frame_index, total_frames), total_frames, time.time() - start_time)

    cap.release()
    writer.release()
    finish_clip()

    # concat helpers
    def concat_group(paths: List[str], out_name: str) -> Optional[str]:
        if not paths:
            return None
        out_path = os.path.join(settings.clips_dir, out_name)
        if ffmpeg_concat_mp4s(paths, out_path):
            return out_path
        return opencv_concat_videos(paths, out_path, fps=fps, size=(scale_w, scale_h))

    made_all = concat_group(made_clip_paths, "made_all.mp4")
    miss_all = concat_group(miss_clip_paths, "miss_all.mp4")

    # Optional compress (best-effort)
    _ = maybe_compress_with_ffmpeg(written_path)
    if made_all:
        _ = maybe_compress_with_ffmpeg(made_all)
    if miss_all:
        _ = maybe_compress_with_ffmpeg(miss_all)

    return {
        "made_total": made_total,
        "missed_total": missed_total,
        "output_video": os.path.basename(written_path),
        "made_clips": [os.path.basename(p) for p in made_clip_paths],
        "miss_clips": [os.path.basename(p) for p in miss_clip_paths],
        "made_all": os.path.basename(made_all) if made_all else None,
        "miss_all": os.path.basename(miss_all) if miss_all else None,
    }
