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
from datasets.transformations import RandomCrop, GreyscaleToRGB
from datasets.modalities import mod_map
from datasets.labels import labels

import os

import numpy as np

import glob
from PIL import Image
from datetime import datetime
from matplotlib import cm  # Color Map


def greyscale_to_rgb(image: torch.Tensor, add_channels_dim=False) -> torch.Tensor:
    """

    :param image:
    :param add_channels_dim:
    :return:
    """
    if add_channels_dim:
        image = image.unsqueeze(-3)

    dims = [-1] * len(image.shape)
    dims[-3] = 3  # RGB, 3 color channels, see explanation below
    return image.expand(*dims)


class Inception3(nn.Module):
    """

    """

    def __init__(self):
        super().__init__()
        # Note - inception_V3 using 229 on 229 images
        self.inception = inception_v3(pretrained=True, transform_input=False, aux_logits=True)
        # aux_logits= True adds an auxiliary branch that can improve training. default =True

        self.inception.fc = nn.Identity()  # FC layer identity replacing FC 1000 features for
        #                                  original inception v3 1000 classes softmax classifier
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
        :param x: a batch of image sequences of either size (NxTxCxHxW) or the squeezed size (NxTxHxW)
        :return: a batch of feature vectors for each image of size (NxTx2048)
        """
        # if the image is greyscale convert it to RGB
        if (len(x.shape) < 5) or (len(x.shape) >= 5 and x.shape[-3] == 1):
            x = greyscale_to_rgb(x, add_channels_dim=len(x.shape) < 5)

        # if we got a batch of sequences we have to calculate each sequence separately
        # n, t = x.shape[:2]  # TODO: understand
        # return self.inception(x.view(-1, *x.shape[2:])).view(n, t, -1)
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
                 RandomCrop((229, 229)), trans.ToTensor(), GreyscaleToRGB()])
        }
    }

    if 'color' in exp_info.modalities_norms.keys():
        modalities['color'] = {
            'max_len': None, 'skip': 1, 'transform': trans.Compose(
                [trans.Normalize(*exp_info.modalities_norms['color']), trans.ToPILImage(),
                 RandomCrop((229, 229)), trans.ToTensor()])
        }

    if 'depth' in exp_info.modalities_norms.keys():
        modalities['depth'] = {
            'max_len': None, 'skip': 1, 'transform': trans.Compose(
                [trans.Normalize(*exp_info.modalities_norms['depth']), trans.ToPILImage(),
                 RandomCrop((229, 229)), trans.ToTensor(), GreyscaleToRGB()])
        }

    modalities.update(
        {
            mod: {
                'max_len': None, 'skip': 1, 'transform': trans.Compose(
                    [trans.Normalize(*norms), trans.ToPILImage(),
                     RandomCrop((458, 458)), trans.ToTensor(), GreyscaleToRGB()])
            } for mod, norms in exp_info.modalities_norms.items() if mod != 'lwir' and mod != 'color'
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
    #  image_path =
    #  ['/home/pnn/experiments/Exp0/2019_06_19_13_50_00_LWIR/2019_06_19_13_50_07_Boson_640_512_Mono16_.tiff']
    #  image_path[0] =
    #  /home/pnn/experiments/Exp0/2019_06_19_13_50_00_LWIR/2019_06_19_13_50_07_Boson_640_512_Mono16_.tiff

    image_path = image_path[0]
    image = Image.open(image_path)
    image = image.crop((left, top, right, bottom))
    image = image.resize((img_len, img_len), resample=Image.NEAREST)

    to_tensor = ToTensor()

    return to_tensor(image).float()


def get_image_dims(file_name: str):
    """

    :param file_name:
    :return:
    """
    fields = file_name.split('/')[-1].split('_')
    return int(fields[8]), int(fields[7])


def get_exposure(file_name: str):
    """

    :param file_name:
    :return:
    """
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

    arr = np.fromfile(image_path, dtype=np.int16).reshape(*get_image_dims(image_path))
    arr = arr[top:bottom, left:right].astype(np.float) / get_exposure(image_path)

    return torch.from_numpy(arr).float().unsqueeze(0)


# def _dir_has_file(self, directory) -> bool:
#     return len(glob.glob(f"{directory}/*{self.vir_type}*.raw")) != 0
#
#
# def color_dir_has_file(directory) -> bool:
#     """
#
#     :rtype: int
#     """
#     return len(glob.glob(directory + '/*.jpg')) != 0


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


def get_depth_image(dire, plant_position, img_len=255):
    """

    :param dire:
    :param plant_position:
    :param img_len:
    :return:
    """
    # 'Depth_day_night'
    left = plant_position[0] - img_len // 2
    right = plant_position[0] + img_len // 2
    top = plant_position[1] - img_len // 2
    bottom = plant_position[1] + img_len // 2

    image_path = glob.glob(dire + '/*ZY_Z16*.raw')
    if len(image_path) == 0:
        raise DirEmptyError()

    image_path = image_path[0]
    img = read_depth_image(image_path, 'uint8', 1280, 1024)  # Image.open(image_path)
    image = Image.fromarray(np.uint8(cm.gist_earth(img) * 255))
    image = image.crop((left, top, right, bottom))

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
            print('mod: ', mod)
            print('===========')
            directory_suffix = mod_map[mod](root_dir, exp_name, **(used_modalities[mod])).directory_suffix

            dirs = sorted(glob.glob(f'{root_dir}/*{directory_suffix}'))
            print('the number of dates in the sorted directories = ', len(dirs))
            print('min  max times for sorted files in mod:', dirs[0].replace(root_dir, ''),
                  dirs[-1].replace(root_dir, ''))
            dirs = filter_dirs(dirs, curr_experiment.start_date.date(), curr_experiment.end_date.date(), root_dir,
                               directory_suffix)
            # print('dirs  = ', dirs)

            for dire in dirs:
                date = get_dir_date_time(dire, root_dir, directory_suffix)
                features = []
                try:
                    for plant in range(num_plants):
                        if mod == 'lwir':
                            image = get_lwir_image(dire, plant_positions[exp_name].lwir_positions[plant])
                        elif mod in ("577nm", "692nm", "732nm ", "970nm", "Polarizer", "PolarizerA"):
                            image = get_vir_image(dire, mod, plant_positions[exp_name].vir_positions[plant])
                        elif mod == "color":
                            image = get_color_image(dire, plant_positions[exp_name].color_positions[plant])
                        elif mod == "depth":
                            image = get_depth_image(dire, plant_positions[exp_name].depth_positions[plant])
                        else:
                            print('unprepared mod =  ', mod)
                            print('  !!!! ')
                        # image shape =  torch.Size([1, 255, 255])
                        # print('color image.shape = ', image.shape)  # torch.Size([3, 254, 254])
                        if image.shape[0] == 1:
                            image = torch.squeeze(image, dim=0)
                            # print('color image.shape = ', image.shape) # torch.Size([3, 254, 254])
                            # image shape =  torch.Size([255, 255])
                            # because I am working with a single image [None,...], and then sending to device
                            image = image[None, ...].to(device)
                        else:
                            image = image.to(device)
                        feature = net(image)
                        features.append(feature)

                except DirEmptyError:
                    print('Empty Directory ', dire)
                    pass
                torch.save(features, root + '_'.join([parameters.experiment, mod, 'features ', str(date)]))


if __name__ == '__main__':
    main()
