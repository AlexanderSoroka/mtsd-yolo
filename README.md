1. Register on mapillary site and download the Dataset

2. Go to the folder with downloaded zip arrays and unpack all of them:
```
cd mapillary
find . -name '*.zip' -exec unzip {} \;
mv images/ mtsd_v2_fully_annotated/
```

3. Setup training virtual environment and activate it
```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
```

4. Setup Ultralytics framework and dependencies
```
pip install ultralytics datumaro PyYAML tqdm
```

5. Convert MTSD to ultralytics yolo format
```
python convert.py --mtsd-root mtsd_v2_fully_annotated --output yolo
```

6. Train YOLO
```
yolo detect train model=yolo11s.pt data=yolo/data.yaml imgsz=720 epochs=200
```