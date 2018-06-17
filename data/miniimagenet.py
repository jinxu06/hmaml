import csv
import glob
import os
import shutil
from PIL import Image

data_dir = "/Users/Aaron-MAC/data/miniimagenet"

def pre_processing():

    print("unfinished, haven't downloaded imagenet yet")
    quit()

    for dataset in ['train', 'val', 'test']:
        all_images = glob.glob(os.path.join(data_dir, dataset, "*", "*"))
        for i, img_file in enumerate(all_images):
            img = Image.open(img_file)
            if not img.size == (84, 84):
                img = img.resize((84, 84), resample=Image.LANCZOS)
                img.save(img_file)
        print(" *** finish processing ", dataset)

    if not os.path.exists(os.path.join(data_dir, 'train')):
        for dataset in ['train', 'val', 'test']:
            os.makedirs(os.path.join(data_dir, dataset))
            with open(os.path.join(data_dir, dataset + ".csv"), 'r') as f:
                reader = csv.reader(f, delimiter=",")
                last_label = ''
                next(reader)
                for i, row in emumerate(reader):
                    label = row[1]
                    img_name = row[0]
                    if label != last_label:
                        cur_dir = os.path.join(data_dir, dataset, label)
                        os.makedirs(cur_dir)
                        last_label = label
                    shutil.move("images/"+img_name, cur_dir)
    else:
        print(" *** image folders already exist")
