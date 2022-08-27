from PIL import ImageEnhance
import json
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


class PostProcessing:
    def __init__(self):
        # Load vietocr
        config = Cfg.load_config_from_name('vgg_transformer')

        # config['weights'] = './weights/transformerocr.pth'
        config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
        config['cnn']['pretrained'] = False
        config['device'] = 'cuda:0'
        config['predictor']['beamsearch'] = False
        self.detector = Predictor(config)

    def crop_info(self, img, list_box):
        crop_img = img.crop((list_box[0]["xmin"], list_box[0]["ymin"]-5, list_box[len(
            list_box)-1]["xmax"], list_box[len(list_box)-1]["ymax"]+5))
        enhancer = ImageEnhance.Sharpness(crop_img)
        factor = 2
        crop_img = enhancer.enhance(factor)
        return crop_img


    def crop_info_one_box(self, img, box):
        crop_img = img.crop(
            (box["xmin"], box["ymin"], box["xmax"], box["ymax"]))
        enhancer = ImageEnhance.Sharpness(crop_img)
        factor = 2
        crop_img = enhancer.enhance(factor)
        return crop_img


    def extract_info(self, img):
        result = self.detector.predict(img)
        return result
