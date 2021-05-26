import os
import torchvision.datasets as datasets
import torch
import glob

phase=['amazon','caltech','dslr','webcam']

dict={'back_pack': 0,
'bike':1,
'calculator':2,
'headphones':3,
'keyboard':4,
'laptop_computer':5,
'monitor':6,
'mouse':7,
'mug':8,
'projector':9}
for i in range(len(phase)):
    path = os.path.join('/data2/gyang/DA-transformer-other/object/data/office_caltech',phase[i])
    text_path=phase[i]+'_list.txt'
    f=open(text_path,'w')

    for label in os.listdir(path):
            img_list = glob.glob(os.path.join(path, label, "*.jpg"))
            for img in img_list:
                    f.write(img + " " + str(dict[label]) + "\n")
    print("create txt done...")