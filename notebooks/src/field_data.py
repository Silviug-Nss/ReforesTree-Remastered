from typing import List, Optional

import pandas as pd

from .datatypes import FieldData


class FieldDataRegistry:
    '''
    This class holds the field measurements dataset, and allow the user to query this dataset by index, or image name
    '''
    def __init__(self, path_to_field_data: str = "./data/raw/field_data.csv"):
        self.path_to_field_data = path_to_field_data
        self.field_data_df = pd.read_csv(self.path_to_field_data)

    def get_field_data_instance(self, index: int) -> FieldData:
        '''
        Get field data by index

        Parameters
        ----------
        index: int
            Index of row in CSV file
        
        Returns
        -------
        FieldData
        '''
        instance = self.field_data_df.iloc[[index]]
        return self._df_row_to_field_data(instance)

    def get_field_data_for_image(self, image_name: Optional[str] = None) -> List[FieldData]:
        '''
        Get field data by image name

        Parameters
        ----------
        image_name: str
            E.g. "Nestor Macias RGB"
        
        Returns
        -------
        List[FieldData]
        '''
        field_data_df = self.field_data_df[self.field_data_df.site == image_name]
        field_data_list = []
        for idx in range(len(field_data_df)):
            row = field_data_df.iloc[[idx]]
            field_data_list.append(self._df_row_to_field_data(row))
        return field_data_list

    @staticmethod
    def _df_row_to_field_data(row: pd.DataFrame) -> FieldData:
        return FieldData(
            name=row.name.values[0],
            group=row.group.values[0],
            lat=row.lat.values[0],
            lon=row.lon.values[0],
            diameter=row.diameter.values[0],
            updated_diameter=row["updated diameter"].values[0],
            height=row.height.values[0],
            year=row.year.values[0],
            plot_id=row.plot_id.values[0],
            site=row.site.values[0],
            x=row.X.values[0],
            y=row.Y.values[0],
            AGB=row.AGB.values[0],
            carbon=row.carbon.values[0]
        )
