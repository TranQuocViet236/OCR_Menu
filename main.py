from model import Model
from post_processing import PostProcessing

MODEl = Model()
MODEl.load_model()
data, img = MODEl.predict("Imgs/051.jpeg")
# print(data)

detector = PostProcessing()
for idx in range(len(data)):
    crop_img = detector.crop_info_one_box(img, data[idx])
    # print(crop_img)
    s = detector.extract_info(crop_img)
    # print(s)
    crop_img.save(f'img_save/img_save_{idx}.png')
    data[idx]['content'] = s

print(data)

