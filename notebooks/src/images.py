import os
from typing import List

import pandas as pd


class ImagesRegistry:
    '''
    This class allows the user to get the following metadata of the Reforestree dataset:
    - Orthomosaic names
    - Tile names
    '''
    def __init__(
        self, annotations_filepath: str = "/workspaces/AI4Good_Group3a/data/raw/annotations/all_annotations.csv"
    ) -> None:
        self.annotation_df = pd.read_csv(annotations_filepath)

    def get_orthomosaic_names(self) -> List[str]:
        return list(self.annotation_df.img_name.unique())

    def get_tile_names(self) -> List[str]:
        tile_names_with_ext = self.annotation_df.img_path.unique()
        tile_names = [os.path.splitext(s)[0] for s in tile_names_with_ext]
        return tile_names
