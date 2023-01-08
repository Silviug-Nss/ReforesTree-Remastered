from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class TreeLabel(Enum):
    TREE = "Tree"
    BANANA = "banana"
    NOT_BANANA = "non-banana"


@dataclass
class BBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class DeepforestDetection(BBox):
    score: float
    label: TreeLabel


@dataclass
class FieldData:
    name: str
    group: str
    lat: float
    lon: float
    diameter: float
    updated_diameter: float
    height: float
    year: int
    plot_id: str
    site: str
    x: float
    y: float
    AGB: float
    carbon: float


@dataclass
class MatchedFieldData(FieldData):
    tree_detection: DeepforestDetection

    @staticmethod
    def to_dict(matched_field_data: "MatchedFieldData") -> Dict[str, Any]:
        return {
            "name": matched_field_data.name,
            "group": matched_field_data.group,
            "lat": matched_field_data.lat,
            "lon": matched_field_data.lon,
            "diameter": matched_field_data.diameter,
            "updated_diameter": matched_field_data.updated_diameter,
            "height": matched_field_data.height,
            "year": matched_field_data.year,
            "plot_id": matched_field_data.plot_id,
            "site": matched_field_data.site,
            "x": matched_field_data.x,
            "y": matched_field_data.y,
            "AGB": matched_field_data.AGB,
            "carbon": matched_field_data.carbon,
            "Xmin": matched_field_data.tree_detection.xmin,
            "Ymin": matched_field_data.tree_detection.ymin,
            "Xmax": matched_field_data.tree_detection.xmax,
            "Ymax": matched_field_data.tree_detection.ymax,
            "score": matched_field_data.tree_detection.score,
            "label": matched_field_data.tree_detection.label,
        }
