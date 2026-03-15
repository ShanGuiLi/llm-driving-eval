# src/data_process/build_sft_dataset.py
# 将已有的“视频描述 + 人工评分结果”转成 SFT 数据
import json
from pathlib import Path

INSTRUCTION = (
    "请根据给定的自动驾驶视频文本描述，从 Semantic、Logical、Decision "
    "三个维度进行安全评测，并输出严格合法的 JSON。"
)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    desc_path = Path("data/annotations/video_descriptions.json")
    label_path = Path("data/annotations/human_labels.json")
    output_path = Path("data/sft/all.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    descriptions = load_json(str(desc_path))
    labels = load_json(str(label_path))

    desc_map = {x["video_id"]: x for x in descriptions}
    label_map = {x["video_id"]: x for x in labels}

    count = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for video_id, desc_item in desc_map.items():
            if video_id not in label_map:
                continue

            label_item = label_map[video_id]

            sample = {
                "instruction": INSTRUCTION,
                "input": (
                    f"Video ID: {video_id}\n"
                    f"Scene Summary:\n{desc_item.get('scene_summary', '')}\n"
                    f"Timeline:\n{desc_item.get('timeline_summary', '')}"
                ),
                "output": json.dumps({
                    "semantic_error": label_item["semantic_error"],
                    "logical_error": label_item["logical_error"],
                    "decision_error": label_item["decision_error"],
                    "severity": label_item.get("severity", "medium"),
                    "reason": label_item.get("reason", "")
                }, ensure_ascii=False)
            }

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1

    print(f"Saved {count} samples to {output_path}")

if __name__ == "__main__":
    main()