from typing import Any, List, Tuple

import pandas as pd


class OrthomosaicGps:
    '''
    This class allows user to get GPS metadata about a given orthomosaics
    '''
    def __init__(
        self,
        ortho_data_filepath: str = "/workspaces/AI4Good_Group3a/data/raw/tiles/ortho_data.csv",
        orthomosaic_name: str = None
    ):
        self.ortho_data = pd.read_csv(ortho_data_filepath)
        self.orthomosaic_name = orthomosaic_name

    def set_orthomosaic_name(self, orthomosaic_name: str):
        self.orthomosaic_name = orthomosaic_name

    def get_lon_lat_to_pixel_ratio(self, use_init: bool = False) -> Tuple[float, float]:
        '''Get longitude per pixel and latitude per pixel ratios'''
        column_index = ['ratio_y', 'ratio_x']
        return self._get_ortho_values(column_index, use_init)

    def get_lon_lat_min(self, use_init: bool = False) -> Tuple[float, float]:
        '''Get longitute and latitude of top left pixel of given orthomosaic'''
        column_index = ['lon_min', 'lat_min']
        return self._get_ortho_values(column_index, use_init)

    def get_lon_lat_max(self, use_init: bool = False) -> Tuple[float, float]:
        '''Get longitute and latitude of bottom right pixel of given orthomosaic'''
        column_index = ['lon_max', 'lat_max']
        return self._get_ortho_values(column_index, use_init)

    def calculate_lon_lat_to_x_y(self, lon: float, lat: float, use_init: bool = True) -> Tuple[int, int]:
        '''Transform given longitude and latitude to pixel coordinates'''
        ratio_x, ratio_y, image_lon_min, image_lat_max = self._get_ortho_data(use_init)
        x = (lon - image_lon_min) / ratio_x
        y = (image_lat_max - lat) / ratio_y
        return int(x), int(y)

    def calculate_x_y_to_lon_lat(self, x: float, y: float, use_init: bool = True) -> Tuple[float, float]:
        '''Transform given pixel coordinates to longitude and latitude'''
        ratio_x, ratio_y, image_lon_min, image_lat_max = self._get_ortho_data(use_init)
        lon = x * ratio_x + image_lon_min
        lat = image_lat_max - y * ratio_y
        return lon, lat

    def _get_ortho_data(self, use_init: bool):
        ratio_y, ratio_x = self.get_lon_lat_to_pixel_ratio(use_init=use_init)
        image_lon_min, _ = self.get_lon_lat_min(use_init=use_init)
        _, image_lat_max = self.get_lon_lat_max(use_init=use_init)
        return ratio_x, ratio_y, image_lon_min, image_lat_max

    def _get_ortho_values(self, column_indices: List[str], use_init: False) -> Any:
        image_ortho_data = self.ortho_data[self.ortho_data.name == self.orthomosaic_name]
        num_values = len(column_indices)
        if use_init:
            column_indices = [s + "_init" for s in column_indices]
        return tuple(image_ortho_data[column_indices].to_numpy().reshape((num_values,)).tolist())
