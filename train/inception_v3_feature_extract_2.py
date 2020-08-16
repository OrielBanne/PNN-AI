"""
inception v3 feature extract
"""

import torch
from torch import nn
from torchvision.models import inception_v3
from torchvision.transforms import ToTensor

from datasets.experiments import ExpInfo, experiments_info, plant_positions
from train import parameters
from typing import Dict
import torchvision.transforms as trans
from datasets.transformations import RandomCrop
from datasets.modalities import mod_map
from datasets.labels import labels
import cv2

import os

import numpy as np

import glob
from PIL import Image
from datetime import datetime


def greyscale_to_rgb(image: torch.Tensor):
    """
    :param image: one channel (b&w)
    :return: image with 3 channels (replicas, standing for color channels)
    """
    print('  G2RGB ', end='')
    image = image.unsqueeze(-3)
    dims = [-1] * len(image.shape)
    dims[-3] = 3  # RGB, 3 color channels, see explanation below
    return image.expand(*dims)


class Inception3(nn.Module):
    """
    input is a tensor of [batch, 299, 299, 3]
    output is a tensor of [batch, 2048]
    """

    def __init__(self):
        super().__init__()
        self.inception = inception_v3(pretrained=True, transform_input=False, aux_logits=True)
        # aux_logits= True adds an auxiliary branch that can improve training. default =True

        self.inception.fc = nn.Identity()  # FC layer identity replacing FC 1000 classes softmax classifier
        self.inception.eval()  # evaluation mode

        for p in self.inception.parameters():
            p.requires_grad = False  # freeze pretrained inception model parameters

    def train(self, mode=True):
        """
        make sure that the inception model stays on eval
        :param mode:
        :return: same net with no training
        """
        return self

    def forward(self, x: torch.Tensor):
        """
        In this version I removed T because it is not relevant when pre running the entire data on inception
        :param x: a batch of image sequences of either size (NxCxHxW) or the squeezed size (NxHxW)
        :return: a batch of feature vectors for each image of size (Nx2048)
        N is batch size N
        T is Time, each image is 1 date_time point
        C is channel
        H - Height
        W - Width
        """
        # if the image is greyscale convert it to RGB
        if (len(x.shape) < 4) or (len(x.shape) >= 4 and x.shape[-3] == 1):
            x = greyscale_to_rgb(x)
            print(' ---> shape is now ', list(x.shape), end=' ')

        return self.inception(x)


def get_modalities_params(exp_info: ExpInfo):
    """
    ExpInfo gives the experiments dates and normalizations
    get ExpInfo information for all modalities
    :param exp_info:
    :return: modalities with all information
    """
    modalities: Dict[str, Dict] = {
        'lwir': {
            'max_len': None, 'skip': 1, 'transform': trans.Compose(
                [trans.Normalize(*exp_info.modalities_norms['lwir']), trans.ToPILImage(),
                 RandomCrop((parameters.lwir_crop, parameters.lwir_crop)), trans.ToTensor()])
        }
    }

    if 'color' in exp_info.modalities_norms.keys():
        modalities['color'] = {
            'max_len': None, 'skip': 1, 'transform': trans.Compose(
                [trans.Normalize(*exp_info.modalities_norms['color']), trans.ToPILImage(),
                 RandomCrop((parameters.color_crop, parameters.color_crop)), trans.ToTensor()])
        }

    if 'depth' in exp_info.modalities_norms.keys():
        modalities['depth'] = {
            'max_len': None, 'skip': 1, 'transform': trans.Compose(
                [trans.Normalize(*exp_info.modalities_norms['depth']), trans.ToPILImage(),
                 RandomCrop((parameters.depth_crop, parameters.depth_crop)), trans.ToTensor()])
        }

    modalities.update(
        {
            mod: {
                'max_len': None, 'skip': 1, 'transform': trans.Compose(
                    [trans.Normalize(*norms), trans.ToPILImage(),
                     RandomCrop((parameters.vir_crop, parameters.vir_crop)), trans.ToTensor()])
            } for mod, norms in exp_info.modalities_norms.items() if mod != 'lwir' and mod != 'color' and mod != 'depth'
        }
    )

    return modalities


def get_used_modalities(modalities, excluded_modalities=None):
    """

    :param modalities:
    :param excluded_modalities:
    :return:
    """
    if excluded_modalities is None:
        excluded_modalities = []
    return {mod: args for mod, args in modalities.items() if mod not in excluded_modalities}


class DirEmptyError(Exception):
    """
    check if directory is empty
    """
    print('empty directory')
    pass


def dir_has_file(directory):
    """
    :param directory: directory to check
    :return: True if there is a file, else False
    """
    if os.listdir(directory):
        return True
    else:
        return False


def get_dir_date(directory, root_dir, directory_suffix):
    """
    :param directory:
    :param root_dir:
    :param directory_suffix: Modality
    :return: datetime date of the directory
    """
    dir_format = f"{root_dir}%Y_%m_%d_%H_%M_%S_{directory_suffix}"
    return datetime.strptime(directory, dir_format).date()


def get_dir_date_time(directory, root_dir, directory_suffix):
    """
    :param directory:
    :param root_dir:
    :param directory_suffix: Modality
    :return: datetime date of the directory
    """
    dir_format = f"{root_dir}%Y_%m_%d_%H_%M_%S_{directory_suffix}"
    string = str(datetime.strptime(directory, dir_format).date()) + '_' + \
             str(datetime.strptime(directory, dir_format).time())
    return string


def filter_dirs(dirs, start_d, end_d, root_dir, directory_suffix):
    """
    check if the sorted directories are within the selected time range, and includes a file
    :param dirs: sorted dirs
    :param start_d:
    :param end_d:
    :param root_dir:
    :param directory_suffix:
    :return: filtered dirs that include image files, per start/end dates
    """
    return [d for d in dirs if start_d <= get_dir_date(d, root_dir, directory_suffix) <= end_d and dir_has_file(d)]


def get_lwir_image(dire, plant_position, img_len=255, plant_crop_len: int = 65):
    """
    :param dire: the specific directory
    :param plant_position: the specific plant position required
    :param img_len: determines the size to be sent to inception
    :param plant_crop_len: determines the size of the plant image in pixels
    :return: an image tensor for the plant
    """
    left = plant_position[0] - plant_crop_len // 2
    right = plant_position[0] + plant_crop_len // 2
    top = plant_position[1] - plant_crop_len // 2
    bottom = plant_position[1] + plant_crop_len // 2

    image_path = glob.glob(dire + '/*.tiff')
    if len(image_path) == 0:
        raise DirEmptyError()
    #  image_path =   ['/home/p....6_.tiff']  image_path[0] = /home/p...6_.tiff

    image_path = image_path[0]

    image = cv2.imread(image_path)  # image = Image.open(image_path) - 3 channels image read
    image = image[top:bottom, left:right]  # image.crop((left, top, right, bottom))
    # image = image.resize((img_len, img_len), resample=Image.NEAREST)
    image = cv2.resize(image, dsize=(img_len, img_len), interpolation=cv2.INTER_CUBIC)  # TODO changed to bicubic -check
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    to_tensor = ToTensor()

    return to_tensor(image).float()


def get_exposure(file_name: str):
    return float(file_name.split('ET')[-1].split('.')[0])


def get_vir_image(dire, mod, plant_position, img_len=510):
    """
    VIR image getter function.
    """
    left = plant_position[0] - img_len // 2
    right = plant_position[0] + img_len // 2
    top = plant_position[1] - img_len // 2
    bottom = plant_position[1] + img_len // 2

    image_path = glob.glob(f"{dire}/*{mod}*.raw")
    if len(image_path) == 0:
        raise DirEmptyError()

    image_path = image_path[0]
    # detecting image size and exposure time
    fields = image_path.split('/')[-1].split('_')
    # exposure_time = image_path.split('ET')[-1].split('.')[0]
    # ------------------------------------------
    arr = np.fromfile(image_path, dtype=np.int16).reshape(int(fields[8]), int(fields[7]))

    arr = arr[top:bottom, left:right].astype(np.float) / get_exposure(image_path)

    image = Image.fromarray(arr)

    to_tensor = ToTensor()
    return to_tensor(image).float()


def get_color_image(dire, plant_position, img_len=255):
    """

    :param dire:
    :param plant_position:
    :param img_len:
    :return:
    """
    # 'D465_Color'
    left = plant_position[0] - img_len // 2
    right = plant_position[0] + img_len // 2
    top = plant_position[1] - img_len // 2
    bottom = plant_position[1] + img_len // 2

    image_path = glob.glob(dire + '/*.jpg')
    if len(image_path) == 0:
        raise DirEmptyError()

    image_path = image_path[0]

    image = Image.open(image_path)
    image = image.crop((left, top, right, bottom))

    to_tensor = ToTensor()

    return to_tensor(image).float()


def read_depth_image(image_path, bit_type, w, h):
    raw_image = np.fromfile(image_path, dtype=bit_type)
    # TODO: Clip image to 34000<value<38000 and normalize the data
    return raw_image[:h * w].reshape(h, w)


def get_depth_image(dire, plant_position, img_len=255, plant_crop_len=100):
    """
    :param plant_crop_len: to cut separate plants
    :param dire: directory
    :param plant_position: position for the plant
    :param img_len: the w and h to be sent out for inception
    :return:
    """
    # 'Depth_day_night'
    left = plant_position[0] - plant_crop_len // 2
    right = plant_position[0] + plant_crop_len // 2
    top = plant_position[1] - plant_crop_len // 2
    bottom = plant_position[1] + plant_crop_len // 2

    image_path = glob.glob(dire + '/*ZY_Z16*.raw')
    if len(image_path) == 0:
        raise DirEmptyError()

    image_path = image_path[0]
    img = read_depth_image(image_path, 'uint8', 1280, 1024)  # Image.open(image_path)
    image = Image.fromarray(np.uint8(img * 255))
    image = image.crop((left, top, right, bottom))
    image = np.asarray(image)
    image = cv2.resize(image, dsize=(img_len, img_len), interpolation=cv2.INTER_CUBIC)
    image = Image.fromarray(image)

    to_tensor = ToTensor()

    return to_tensor(image).float()


def main():
    """

    :return:
    """
    exp_name = parameters.experiment
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    curr_experiment = experiments_info[exp_name]
    root_dir = parameters.experiment_path
    modalities = get_modalities_params(curr_experiment)
    used_modalities = get_used_modalities(modalities, parameters.excluded_modalities)

    print('curr_experiment ')
    print(curr_experiment)
    print('used modalities keys = ', used_modalities.keys())
    print('device = ', device)
    print(curr_experiment.start_date.date())
    print(curr_experiment.end_date.date())

    # Saving all features to root
    root = '/home/pnn/inception_features/'

    num_plants = len(labels[exp_name])
    net = Inception3()
    net = net.to(device)
    with torch.no_grad():
        for mod in used_modalities:
            print('\nmod: ', mod)
            print('===========')
            directory_suffix = mod_map[mod](root_dir, exp_name, **(used_modalities[mod])).directory_suffix
            print('Got directory suffix inside pre inception 2')
            dirs = sorted(glob.glob(f'{root_dir}/*{directory_suffix}'))
            dirs = filter_dirs(dirs, curr_experiment.start_date.date(), parameters.end_date.date(), root_dir,
                               directory_suffix)

            for dire in dirs:
                date = get_dir_date_time(dire, root_dir, directory_suffix)
                features = []
                try:
                    for plant in range(num_plants):
                        print('\n plant  ', plant, end='  ')
                        if mod == 'lwir':
                            image = get_lwir_image(dire, plant_positions[exp_name].lwir_positions[plant])
                            image = image.unsqueeze(0)
                            print('Modality ', mod, 'image shape : ', list(image.shape), end='')
                        elif mod in ("577nm", "692nm", "732nm", "970nm", "Polarizer", "PolarizerA"):
                            image = get_vir_image(dire, mod, plant_positions[exp_name].vir_positions[plant])
                            # image = image.unsqueeze(0)
                            print('Modality ', mod, 'image shape is: ', list(image.shape), end='')
                        elif mod == "color":
                            image = get_color_image(dire, plant_positions[exp_name].color_positions[plant])
                            image = image.unsqueeze(0)
                            print('Modality ', mod, 'image shape is: ', list(image.shape), end='')
                        elif mod == "depth":
                            image = get_depth_image(dire, plant_positions[exp_name].depth_positions[plant])
                            print('Modality ', mod, 'image shape is: ', list(image.shape), end='')
                        else:
                            print('unprepared mod =  ', mod)
                            print('  !!!!  ')

                        if image.shape[0] == 1:
                            image = torch.squeeze(image, dim=0)
                            # because I am working with a single image [None,...], and then sending to device
                            image = image[None, ...].to(device)
                            print(' ', mod, 'shape after = ', list(image.shape), end='')
                        else:
                            image = image.to(device)
                        feature = net(image)
                        torch.save(feature, root + '_'.join([parameters.experiment, mod, 'plant', str(plant), str(date)]))
                        features.append(feature)
                        size = feature.element_size() * feature.nelement()
                        print(' size ', size, end='  ')
                        if size < 8192:
                            print('PROBLEM ELEMENT - size is ', size)

                except DirEmptyError:
                    print('Empty Directory ', dire)
                    pass
                # torch.save(features, root + '_'.join([parameters.experiment, mod, 'features ', str(date)]))


if __name__ == '__main__':
    main()
