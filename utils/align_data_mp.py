import sys
sys.path.append('/home/chenyuheng/PTI/')

from configs import paths_config
import dlib
import glob
import os
from tqdm import tqdm
from utils.alignment import align_face

import multiprocessing as mp
from tqdm import tqdm

raw_images_path='/home_new/chenyuheng/ffhq512/512/'
predictor = dlib.shape_predictor(paths_config.dlib)


def pre_process_images(image_name):
    # current_directory = os.getcwd()

    IMAGE_SIZE = 1024


    # aligned_images = []

    try:
        aligned_image = align_face(filepath=f'{raw_images_path}/{image_name}',
                                    predictor=predictor, output_size=IMAGE_SIZE)
    except Exception as e:
        print(e)
        return

    # for image, name in zip(aligned_images, images_names):
    real_name = image_name.split('.')[0]
    aligned_image.save(f'{raw_images_path}/PTI/{real_name}.jpeg')

    # os.chdir(current_directory)


if __name__ == "__main__":
    os.chdir(raw_images_path)
    images_names = glob.glob(f'*')
    os.makedirs(os.path.join(raw_images_path, 'PTI'), exist_ok=True)
    
    with mp.Pool(14) as pool:
        print(list(tqdm(pool.imap(pre_process_images, images_names), total=70000)))
