from PIL import Image
import glob
import os

data_dir = "/Users/Aaron-MAC/data/omniglot"

def pre_processing():
    for dataset in ['images_background', "images_evaluation"]:
        all_images = glob.glob(os.path.join(data_dir, dataset, "*", "*", "*"))
        for i, img_file in enumerate(all_images):
            img = Image.open(img_file)
            if not img.size == (28, 28):
                img = img.resize((28, 28), resample=Image.LANCZOS)
                img.save(img_file)
        print(" *** finish processing ", dataset)
