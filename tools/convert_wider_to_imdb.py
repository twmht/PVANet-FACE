from wider import WIDER
import json
import matplotlib.pyplot as plt
from PIL import Image as pil_image
from io import BytesIO
import struct
from image_format_pb2 import Image

wider = WIDER('/home/tumh/python-wider/data/v1',
              '/home/tumh/python-wider/data/WIDER_train/images',
              'wider_face_train.mat')

image_db = open('image.db', 'wb')
index_to_image_db = open('index.image.db', 'wb')

def write_byte(f, index_file, byte):
    index = f.tell()
    end = len(byte)
    f.write(byte)
    index_file.write(struct.pack('QQ', index, end))


# press ctrl-C to stop the process
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
    write_byte(image_db, index_to_image_db, byte)

image_db.close()
index_to_image_db.close()
