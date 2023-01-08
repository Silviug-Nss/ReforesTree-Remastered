import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

IMAGE_NAME_MAPPING = {
    "Carlos Vera Arteaga RGB": "Vera Arteaga Carlos",
    "Carlos Vera Guevara RGB": "Vera Gevara Carlos",
    "Flora Pluas RGB": "Flora Puas 2017",
    "Leonor Aspiazu RGB": "Aspiazu Mendoza leonor",
    "Manuel Macias RGB": "Macias Guevara Manuel",
    "Nestor Macias RGB": "WWF Nestor Macias",
}


class SiteShape:
    def __init__(self, shape_filepath: str = "./data/raw/wwf_ecuador/Merged_final_plots/Merged_final_plots.shp"):
        '''This class contains data about the region of interest of each site'''
        shapefile = gpd.read_file(shape_filepath)
        self.shapefile = pd.DataFrame(shapefile)
        self.dilation = 0

    def is_in_site(self, orthomosaic_name: str, lon: float, lat: float) -> bool:
        '''Determines where given GPS coords is within the region of interest of given orthomosaic'''
        orthomosaic_name = IMAGE_NAME_MAPPING[orthomosaic_name]
        image_shapefile = self.shapefile[self.shapefile.Name == orthomosaic_name]
        image_site_polygon: Polygon = image_shapefile.geometry.iloc[0]
        site_polygon_buffer = image_site_polygon.buffer(self.dilation)
        point = Point(lon, lat, 0)
        return point.within(image_site_polygon) or point.within(site_polygon_buffer)

    def set_dilation(self, dilation: int):
        '''Set amount of dilation to apply to region of interest'''
        self.dilation = dilation
