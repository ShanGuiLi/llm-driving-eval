from string import Template

VIDEO_DESCRIPTION_PROMPT = Template(
    r"""
You are a strict visual analyst for autonomous driving videos.

The provided images are key frames sampled from the SAME generated driving video.
They are ordered chronologically from earlier to later.

Your task is to describe the video based only on these frames.

Focus only on facts visible in the frames.
Do not guess uncertain content.
Do not mention anything outside the visual evidence.

You must describe the video from three aspects:

1. Semantic facts
- key vehicles
- pedestrians
- traffic lights / road signs
- lane lines / stop lines
- road structure
- obvious missing / wrong / replaced objects

2. Logical facts
- whether objects appear/disappear suddenly
- whether positions jump unnaturally
- whether motion / trajectory looks physically unreasonable
- whether scene continuity is broken

3. Decision facts
- whether ego vehicle behavior seems unsafe
- should stop but does not stop
- should yield but does not yield
- dangerous lane change / dangerous maneuver
- no slowing down in a clearly high-risk situation

Output requirements:
- Return plain text only
- Be concise but specific
- Mention uncertainty explicitly if visual evidence is unclear
- Do not output JSON

Input:
video_id: $video_id

Now describe the generated video based on the provided key frames only.
"""
)