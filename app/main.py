import argparse
import json
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO
from torchvision.models import ResNet

from configure import load_classifier, load_detector
from utils.processing import process_video


def read_meta(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as file:
            meta = json.load(file)
    except:
        meta = None
    return meta


def trigger(video_path: str,
            output_path: str,
            actors_priority_path: str,
            video_metadata_path: str,
            detector: YOLO,
            classifier: ResNet) -> None:
    video_metadata = read_meta(video_metadata_path)
    actors_priority = read_meta(actors_priority_path)

    final_output = process_video(video_metadata=video_metadata,
                                 detector=detector,
                                 classifier=classifier,
                                 actors_priorty=actors_priority,
                                 input_video_path=video_path)
    print(final_output)
    with open(output_path, "w") as f:
        json.dump(final_output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str)
    parser.add_argument('--actors_file', type=str)
    parser.add_argument('--meta_file', type=str)
    args = parser.parse_args()

    input_video_path = str(Path(__file__).parent.parent / "data" / args.video_file)
    actors_priority = str(Path(__file__).parent.parent / "data" / args.actors_file)
    video_metadata = str(Path(__file__).parent.parent / "data" / args.meta_file)

    output_json_path = str(Path("result") / input_video_path.replace(".mp4", "_result.json"))

    classifier = load_classifier(str(Path(__file__).parent.parent / "models" / "classification_model.pth"))
    detector = load_detector()
    trigger(video_path=input_video_path,
            output_path=output_json_path,
            actors_priority_path=actors_priority,
            video_metadata_path=video_metadata,
            detector=detector,
            classifier=classifier)
