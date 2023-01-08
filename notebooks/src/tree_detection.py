import json
import os
from typing import Dict, List

import pandas as pd

from .datatypes import DeepforestDetection, TreeLabel


class DeepForestDetectionRegistry:
    '''This class loads tree detections predicted by deepforest'''
    def __init__(
        self, deepforest_detection_dir: str = "/workspaces/AI4Good_Group3a/data/predicted_bbox"
    ):
        self.deepforest_detection_dir = deepforest_detection_dir

    def load_detections_for_image(self, orthomosaic_name: str) -> List[DeepforestDetection]:
        '''Load detections for given orthomosaic'''
        detections_filename = f"{orthomosaic_name}_predicted_bounding_boxes.csv"
        detections_filepath = os.path.join(self.deepforest_detection_dir, detections_filename)
        detections_df = pd.read_csv(detections_filepath)
        return DeepForestDetectionRegistry._dataframe_to_deepforest_detections(detections_df)

    @staticmethod
    def _dataframe_to_deepforest_detections(detections_df: pd.DataFrame) -> List[DeepforestDetection]:
        detections = []
        for _, row in detections_df.iterrows():
            label = TreeLabel(row.label)
            detection = DeepforestDetection(
                row.xmin,
                row.ymin,
                row.xmax,
                row.ymax,
                row.score,
                label,
            )
            detections.append(detection)
        return detections


class HandAnnotatedDetectionRegistry:
    '''This class loads hand annotated tree bounding boxes'''
    def __init__(self, annotations_filepath: str = "/workspaces/AI4Good_Group3a/data/bboxes"):
        self.annotations_filepath = annotations_filepath

    def load_detections_for_image(self, orthomosaic_name: str) -> List[DeepforestDetection]:
        '''Load detections (hand annotated) for given orthomosaic'''
        orthomosaic_name = orthomosaic_name.replace(" ", "_")[:-4].lower()
        annotations_filename = f"{orthomosaic_name}.json"
        annotations_filepath = os.path.join(self.annotations_filepath, annotations_filename)
        with open(annotations_filepath) as f:
            annotations_file = json.load(f)
        detections_df = pd.DataFrame.from_dict(annotations_file["boxes"])
        return HandAnnotatedDetectionRegistry._dataframe_to_deepforest_detections(detections_df)

    @staticmethod
    def _dataframe_to_deepforest_detections(detections_df: pd.DataFrame) -> List[DeepforestDetection]:
        detections = []
        for _, row in detections_df.iterrows():
            label = TreeLabel(row.label)
            x, y, w, h = float(row.x), float(row.y), float(row.width), float(row.height)
            xmin = int(x - w/2)
            ymin = int(y - h/2)
            xmax = int(x + w/2)
            ymax = int(y + h/2)
            score = 1 if row.label == "banana" else 0
            detection = DeepforestDetection(
                xmin,
                ymin,
                xmax,
                ymax,
                score,
                label,
            )
            detections.append(detection)
        return detections