import os
import glob
from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed

ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize_image(image_path, output_folder, resize):
    base_name = os.path.basename(image_path)
    output = os.path.join(output_folder, base_name)
    img = Image.open(image_path)
    img = img.resize((resize[1], resize[0]), resample=Image.BILNEAR)
    img.save(outpath)


os.mkdir('/kaggle/working/train512')
os.mkdir('/kaggle/woeking/test512') 

input_folder = '../input/siim-isic-meloanoma-classification/jpeg/train/'
output_folder = '/kaggle/working/train512'
images = glob.glob(os.path.join(input_folder, '*.jpg'))

Parallel(n_jobs=12)(
    delayed(resize_image)(
        i,
        output_folder,
        [512,512]
    ) for i in tqdm(images)
)

input_folder = '../input/siim-isic-meloanoma-classification/jpeg/test/'
output_folder = '/kaggle/working/test512'
images = glob.glob(os.path.join(input_folder, '*.jpg'))

Parallel(n_jobs=12)(
    delayed(resize_image)(
        i,
        output_folder,
        [512,512]
    ) for i in tqdm(images)
)