import os
import shutil
import numpy as np
import jpegio as jpio

def move(src, dst):
    if not os.path.isdir(dst):
        os.makedirs(dst)
    shutil.copy(src, dst)

# img_dirs = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
img_dirs = ['Test']
for img_dir in img_dirs:
    for i, img in enumerate(os.listdir('/home/shareData/ALASKA2/' + img_dir)):
        source = os.path.join('/home/shareData/ALASKA2/' , img_dir , img)
        jpegStruct = jpio.read(source)
        if (jpegStruct.quant_tables[0][0,0]==2):
            print('Quality Factor is 95')
            dst = os.path.join('/home/shareData/ALASKA2/QF95' , img_dir , img)
            move(source,dst)
        elif (jpegStruct.quant_tables[0][0,0]==3):
            print('Quality Factor is 90')
            dst = os.path.join('/home/shareData/ALASKA2/QF90' , img_dir , img)
            move(source,dst)
        elif (jpegStruct.quant_tables[0][0,0]==8):
            print('Quality Factor is 75')
            dst = os.path.join('/home/shareData/ALASKA2/QF75', img_dir, img)
            move(source, dst)