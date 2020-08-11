########################################################################################################
#                                                                                                      #
#                                         ModalityDataset                                              #
#                                                                                                      #
########################################################################################################


import glob
from abc import abstractmethod

import torch
from torch.utils import data
from torchvision import transforms
from typing import Tuple

from datasets.labels import labels
from datasets.transformations import RandomPNNTransform
from train import parameters
from datetime import datetime


# just checking DirEmptyError
class DirEmptyError(Exception):
    pass


class ModalityDataset(data.Dataset):
    """
    The parent class for datasets from the experiment.
    """
    def __init__(self, root_dir: str, exp_name: str, directory_suffix: str, img_len: int,
                 positions: Tuple[Tuple[int, int], ...], split_cycle=parameters.split_cycle,
                 start_date=parameters.start_date, end_date=parameters.end_date, skip=1, max_len=None, transform=None):
        """
        :param root_dir: path to the experiment directory
        :param exp_name: the experiment we want to use
        :param directory_suffix: the directory name suffix for the image type # TODO where does it come from??
        :param positions: the positions of the plants within the images
        :param img_len: the length that the images will be resized to
        :param split_cycle: amount of days the data will be split by
        :param skip: how many frames to skip between ones taken
        :param max_len: the max amount of images to use; if None - no limit
        :param transform: optional transform to be applied on each frame
        """
        print('\n ModalityDataset Class')
        print('root_dir=', root_dir)

        self.root_dir = root_dir

        print('dir_suffix=  ', directory_suffix)
        self.directory_suffix = directory_suffix
        dirs = sorted(glob.glob(f'{root_dir}/*{directory_suffix}'))
        # dd = [d.replace(root_dir, '') for d in dirs]
        # dd = [d.replace(directory_suffix, '') for d in dd]
        # # print('sorted directories = ', dd)
        print('the number of dates in the sorted directories = ', len(dirs))
        print('min  max times for sorted files in mod:', dirs[0].replace(root_dir, ''),
              dirs[-1].replace(root_dir, ''))
        dirs = self.__filter_dirs(dirs, start_date.date(), end_date.date())
        # dd = [d.replace(root_dir, '') for d in dirs]
        # dd = [d.replace(directory_suffix, '') for d in dd]
        # # print('filtered dirs = ', dd)
        print('the number of dates in the filtered directories = ', len(dirs))
        self.cycles_dirs = self.__get_cycles_dirs(dirs, split_cycle, skip)

        self.exp_name = exp_name
        self.positions = positions

        self.num_plants = len(positions)

        self.img_len = img_len
        self.split_cycle = split_cycle

        highest_max_len = min(len(cycle_dirs) for cycle_dirs in self.cycles_dirs)
        if max_len is None:
            self.max_len = highest_max_len
        else:
            self.max_len = min(max_len, highest_max_len)

        self.cycles_dirs = tuple(tuple(cycle_dirs[:self.max_len]) for cycle_dirs in self.cycles_dirs)

        if transform is None:
            self.transform = transforms.Compose([])
        else:
            self.transform = transform

    def __get_dir_date(self, directory):
        dir_format = f"{self.root_dir}%Y_%m_%d_%H_%M_%S_{self.directory_suffix}"
        return datetime.strptime(directory, dir_format).date()

    def __get_cycles_dirs(self, dirs, split_cycle, skip):
        curr_date = self.__get_dir_date(dirs[0])
        days_dirs = [[]]

        for directory in dirs:
            # update the current day when it changes
            dir_date = self.__get_dir_date(directory)
            if curr_date != dir_date:
                curr_date = dir_date
                days_dirs.append([])

            # get the image only every split_cycle days
            days_dirs[-1].append(directory)

        cycles_dirs = [sum(days_dirs[idx::split_cycle], []) for idx in range(split_cycle)]
        cd_for_printout = [cycle_dirs[::skip] for cycle_dirs in cycles_dirs]
        print('min  max times for mod:', cd_for_printout[0][0].replace(self.root_dir, ''),
              cd_for_printout[0][-1].replace(self.root_dir, ''))
        return [cycle_dirs[::skip] for cycle_dirs in cycles_dirs]

    def __filter_dirs(self, dirs, start_d, end_d):
        return [d for d in dirs if start_d <= self.__get_dir_date(d) <= end_d and self._dir_has_file(d)]

    def __len__(self):
        return self.num_plants * self.split_cycle

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError()

        # the day in the cycle this sample belongs to
        cycle_day = idx // self.num_plants
        plant = idx % self.num_plants

        tensors = []

        for directory in self.cycles_dirs[cycle_day]:
            try:
                image = self._get_image(directory, self.positions[plant])
                tensors.append(image)
            except DirEmptyError:
                print('----------------------------------------------------------------')
                print('Empty Directory ', directory)
                print('----------------------------------------------------------------')
                pass

        for t in self.transform.transforms:
            if isinstance(t, RandomPNNTransform):
                t.new_params()

        image = torch.stack([self.transform(tensor) for tensor in tensors])

        sample = {'image': image, 'label': labels[self.exp_name][plant],
                  'position': self.positions[plant], 'plant': plant}

        return sample

    @abstractmethod
    def _get_image(self, directory, plant_position):
        pass

    @abstractmethod
    def _dir_has_file(self, directory) -> bool:
        pass
