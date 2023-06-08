import os
from glob import glob
import shutil
import numpy as np
import pandas as pd

df = pd.read_excel('./class.xlsx', index_col=0)
print(df[df.duplicated(['file_name'])])
for idx, row in df.iterrows():
    scr = f"./segment_bakcha/segment/{row.file_name}"
    if row.pred == 'front':
        dst = f"./segment_bakcha/front/{row.file_name}"
    else:
        dst = f"./segment_bakcha/side/{row.file_name}"
    shutil.copy(scr, dst)