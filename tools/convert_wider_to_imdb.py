from PIL import Image as pil_image
from cute.cute_writer import CuteWriter
from image_format_pb2 import Image
from io import BytesIO
from wider import WIDER
import json
import struct

def run(wider, cw):
    for data in wider.next():
        f_name = data.image_name.split('/')[-1].split('.')[0]
        print f_name


        if f_name =='0_Parade_Parade_0_452':
            continue
        if f_name =='2_Demonstration_Political_Rally_2_444':
            continue
        if f_name =='39_Ice_Skating_iceskiing_39_380':
            continue
        if f_name =='46_Jockey_Jockey_46_576':
            continue

        im = pil_image.open(data.image_name)
        width, height = im.size
        mem = BytesIO()
        im.save(mem, 'JPEG')
        mem.seek(0)
        # write image to db
        image = Image()
        image.name = data.image_name
        image.data = mem.read()
        image.width = width
        image.height = height

        for bbox in data.bboxes:
            xmin=bbox[0]
            ymin=bbox[1]
            xmax=bbox[2]
            ymax=bbox[3]

            if xmax - xmin < 3 or ymax - ymin < 3:
                continue
            if xmin < 1:
                xmin = 1
            if xmax >= width:
                xmax = width-2
            if ymin < 1:
                ymin = 1
            if ymax >= height:
                ymax = height - 2
            if xmin >= xmax or ymin >= ymax:
                continue


            obj = image.objects.add()
            obj.minX = xmin
            obj.minY = ymin
            obj.maxX = xmax
            obj.maxY = ymax


        if len(image.objects) == 0:
            continue
        byte = image.SerializeToString()
        cw.write(byte)


wider_train = WIDER('/opt/WiderFace/wider_face_split',
              '/opt/WiderFace/WIDER_train/images',
              'wider_face_train.mat')

cw = CuteWriter('wider-imdb')

run(wider_train, cw)

cw.close()
