import os
from typing import List

import cv2


def _resize_to_qwen25vl_safe(frame, max_side: int = 672):
    """
    把图像缩放到较安全的尺寸，并对齐到 28 的倍数。
    qwen2.5vl 在 Ollama 里对图像尺寸比较敏感。
    """
    h, w = frame.shape[:2]

    # 按最长边缩放
    scale = min(max_side / max(h, w), 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 对齐到 28 的倍数，且至少 28
    new_w = max(28, (new_w // 28) * 28)
    new_h = max(28, (new_h // 28) * 28)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def extract_key_frames(
    video_path: str,
    output_dir: str,
    num_frames: int = 6
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Invalid total frame count: {video_path}")

    if num_frames == 1:
        indices = [total_frames // 2]
    else:
        indices = [
            int(i * (total_frames - 1) / (num_frames - 1))
            for i in range(num_frames)
        ]

    frame_paths = []
    saved = set()

    for idx in indices:
        if idx in saved:
            continue
        saved.add(idx)

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        frame = _resize_to_qwen25vl_safe(frame, max_side=672)

        frame_name = f"frame_{idx:06d}.jpg"
        frame_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_paths.append(frame_path)

    cap.release()

    if not frame_paths:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return frame_paths