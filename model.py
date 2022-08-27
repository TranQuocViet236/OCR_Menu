from PIL import Image, ImageOps
from extract_info import InfoExtractModel
from post_processing import PostProcessing
from PIL import ImageEnhance



class Model:
    def __init__(self):
        self.info_model = None
        # self.post_processing = PostProcessing()


    def load_model(self):
        path_weight_info = 'weight/best.pt'
        self.info_model = InfoExtractModel(path_weight=path_weight_info)



    def predict(self, img_path):
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)
        data = self.info_model.info_predict(img)

        return data, img






