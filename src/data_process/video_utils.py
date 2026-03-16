import os
from typing import List

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def _align_to_multiple(value: int, base: int = 28, min_value: int = 28) -> int:
    aligned_value = max(min_value, (value // base) * base)
    return aligned_value


def _resize_to_qwen25vl_safe(frame_array: np.ndarray, max_side: int = 672) -> Image.Image:
    """
    Resize the frame to a safer size for Qwen2.5-VL and align width and height
    to multiples of 28.
    """
    if frame_array.ndim == 2:
        image = Image.fromarray(frame_array).convert("RGB")
    elif frame_array.ndim == 3:
        if frame_array.shape[2] == 4:
            image = Image.fromarray(frame_array).convert("RGB")
        else:
            image = Image.fromarray(frame_array[:, :, :3]).convert("RGB")
    else:
        raise ValueError(f"Unsupported frame shape: {frame_array.shape}")

    width, height = image.size

    scale = min(max_side / max(width, height), 1.0)
    new_width = int(width * scale)
    new_height = int(height * scale)

    new_width = _align_to_multiple(new_width, base=28, min_value=28)
    new_height = _align_to_multiple(new_height, base=28, min_value=28)

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image


def _get_total_frames(reader, video_path: str) -> int:
    """
    Try to get total frame count from the reader. Fall back to metadata if needed.
    """
    try:
        total_frames = reader.count_frames()
        if total_frames and total_frames > 0:
            return int(total_frames)
    except Exception:
        pass

    try:
        metadata = reader.get_meta_data()
        total_frames = metadata.get("nframes", 0)
        if total_frames and total_frames > 0 and total_frames != float("inf"):
            return int(total_frames)
    except Exception:
        pass

    raise ValueError(f"Cannot determine total frame count: {video_path}")


def _build_frame_indices(total_frames: int, num_frames: int) -> List[int]:
    if total_frames <= 0:
        raise ValueError("Total frame count must be positive")

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    if num_frames == 1:
        return [total_frames // 2]

    indices = [
        int(i * (total_frames - 1) / (num_frames - 1))
        for i in range(num_frames)
    ]

    unique_indices = []
    visited = set()
    for index in indices:
        if index not in visited:
            visited.add(index)
            unique_indices.append(index)

    return unique_indices


def extract_key_frames(
    video_path: str,
    output_dir: str,
    num_frames: int = 16
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    reader = None
    try:
        reader = imageio.get_reader(video_path, format="ffmpeg")
        total_frames = _get_total_frames(reader, video_path)
        frame_indices = _build_frame_indices(total_frames, num_frames)

        frame_paths: List[str] = []

        for frame_index in frame_indices:
            try:
                frame_array = reader.get_data(frame_index)
            except Exception:
                continue

            resized_image = _resize_to_qwen25vl_safe(frame_array, max_side=672)

            frame_name = f"frame_{frame_index:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            resized_image.save(frame_path, format="JPEG", quality=95)
            frame_paths.append(frame_path)

        if not frame_paths:
            raise ValueError(f"No frames extracted from video: {video_path}")

        return frame_paths

    finally:
        if reader is not None:
            reader.close()