import os
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from lang_sam import LangSAM

from classifier import ShotAngleClassifier

img_paths = glob('./segment_bakcha/segment_bakcha/*.png')  # 내 드라이브 이미지 경로 입력

model = LangSAM()
text_prompt = "tire wheel"
angle_threshold = 10
circle_threshold = 0.85

sac = ShotAngleClassifier()
table = pd.DataFrame(np.empty((len(img_paths), 2)), columns=['file_name', 'label'], dtype='str')


for idx, path in tqdm(enumerate(img_paths[:])):
    idx += 0
    image_pil = Image.open(path).convert("RGB")
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)

    masks_arr = np.zeros(masks.shape, dtype='uint8')
    for n, mask in enumerate(masks):
        mask_arr = mask.detach().cpu().numpy()
        y, x = np.where(mask_arr == True)
        masks_arr[n, y, x] = 255

    bool_label = sac.front_or_not(
        masks_arr, boxes, logits, angle_threshold, circle_threshold
    )

    if bool_label:
        table.iloc[idx] = {'file_name': os.path.basename(path), 'label': 'front'}
    else:
        table.iloc[idx] = {'file_name': os.path.basename(path), 'label': 'side'}

table.to_excel('pred.xlsx', index=False)
