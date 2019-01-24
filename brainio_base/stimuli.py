import pandas as pd


class StimulusSet(pd.DataFrame):
    _metadata = ["get_image", "image_paths"]

    @property
    def _constructor(self):
        return StimulusSet

    def get_image(self, image_id):
        return self.image_paths[image_id]
