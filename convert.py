#!/usr/bin/env python3
import argparse
import os
import json
import yaml
import tqdm

from datumaro import Dataset, DatasetItem
from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories
from datumaro.components.media import Image as DmImage


def read_split_list(path: str) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def write_dataset_yaml(out_dir: str, subsets: list[str], class_names: list[str]) -> None:
    data = {s: str((out_dir / "images" / s).resolve()) for s in ["train", "val", "test"] if s in subsets}
    data["names"] = class_names
    data["nc"] = len(class_names)
    with (out_dir / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main():
    args = argparse.ArgumentParser("Convert MTSD -> YOLO (Ultralytics)")
    args.add_argument("--mtsd-root", type=str, required=True)
    args.add_argument("--output", type=str, required=True)
    args.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    args.add_argument("--img-ext", default=".jpg")
    args = args.parse_args()

    args.mtsd = args.mtsd_root

    if os.path.exists(args.output):
        print('Failed to create output directory: output path exists')
        exit(1)
    
    os.mkdir(args.output)
    
    images_dir = os.path.join(args.mtsd, 'images')
    annotation_dir = os.path.join(args.mtsd, 'annotations')
    splits_dir = os.path.join(args.mtsd, 'splits')

    all_labels = set()
    split_keys = {}
    
    for split_name in args.splits:
        file_ids = read_split_list(os.path.join(splits_dir, f"{split_name}.txt"))
        split_keys[split_name] = file_ids
        for file_id in file_ids:
            annotation_path = os.path.join(annotation_dir, f"{file_id}.json")
            if not os.path.exists(annotation_path.exists):
                continue

            with open(annotation_path, "r", encoding="utf-8") as annotation_input:
                annotation = json.load(annotation_input)

            for object in annotation.get("objects", []):
                all_labels.add(str(object["label"]))

    label_categories = LabelCategories()
    for name in sorted(all_labels):
        label_categories.add(name)

    ds = Dataset(media_type=DmImage)
    ds.define_categories({AnnotationType.label: label_categories})

    # populate items (skip missing annotation files; keep image)
    for subset, file_ids in split_keys.items():
        print(f'Processing subset: {subset}')
        for file_id in tqdm.tqdm(file_ids):
            media = DmImage.from_file(os.path.join(images_dir, f"{file_id}{args.img_ext}"))
            annotation_path = os.path.join(annotation_dir, f"{file_id}.json")
            annotations = []
            if os.path.exists(annotation_path):
                with open(annotation_path, "r", encoding="utf-8") as annotation_input:
                    annotation = json.load(annotation_input)

                for object in annotation.get("objects", []):
                    bounding_box = object["bbox"]
                    x1, y1 = float(bounding_box["xmin"]), float(bounding_box["ymin"])
                    x2, y2 = float(bounding_box["xmax"]), float(bounding_box["ymax"])
                    w, h = x2 - x1, y2 - y1
                    label_idx = label_categories.find(str(object["label"]))[0]
                    annotations.append(Bbox(x1, y1, w, h, label=label_idx))

            ds.put(DatasetItem(id=file_id, media=media, annotations=annotations, subset=subset))

    # Export to Ultralytics layout
    ds.export(save_dir=str(args.output), format="yolo_ultralytics", save_media=True)

    # dataset.yaml for Ultralytics
    write_dataset_yaml(args.output, args.splits, [c.name for c in label_categories])

    print(f"[DONE] Exported YOLO (Ultralytics) dataset to {args.output}")


if __name__ == "__main__":
    main()
