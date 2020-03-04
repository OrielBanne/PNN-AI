
from datetime import datetime
import glob
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms

from .labels import labels
from .exceptions import *


# plants are indexed left to right, top to bottom - Exp0
Exp0positions = [
    (146, 105), (206, 100), (265, 97), (322, 98),
    (413, 105), (464, 105), (517, 110), (576, 115),
    (149, 157), (212, 152), (262, 145), (320, 142),
    (416, 167), (468, 165), (522, 169), (575, 171),
    (155, 207), (213, 205), (264, 204), (322, 200),
    (417, 213), (467, 218), (522, 216), (573, 219),
    (157, 263), (212, 261), (267, 258), (321, 260),
    (418, 266), (470, 266), (528, 263), (574, 270),
    (156, 317), (212, 315), (265, 315), (327, 319),
    (418, 321), (468, 314), (522, 314), (574, 319),
    (154, 366), (215, 368), (269, 372), (326, 374),
    (417, 373), (465, 375), (520, 373), (573, 369)
]

# plants are indexed left to right, top to bottom - Exp3
Exp3positions = [
	(86, 108), (186, 100), (294, 98),  (388, 100), (484, 104), (566, 100),
	(84, 194), (188, 194), (300, 196), (398, 198), (486, 198), (572, 198),
	(86, 278), (198, 282), (302, 282), (398, 290), (488, 286), (570, 290),
	(88, 360), (198, 362), (304, 364), (404, 364), (484, 356), (576, 358),
	(88, 434), (204, 430), (316, 440), (406, 440), (488, 432), (576, 432)
]


## here choose positions and dates tocorrect experiment
positions = Exp0positions

## define datetime per experiment:

Exp0_dates = [(2019, 6,  5),(2019, 7, 7)]
Exp1_dates = [(2019, 7, 28),(2019, 8, 4)]
Exp2_dates = [(2019, 9, 20),(2019, 10, 13)]
Exp3_dates = [(2019, 10, 28),(2019, 12, 7)]

#for Exp0
start_date,end_date = Exp0_dates


class LWIR(data.Dataset):
    def __init__(self, root_dir: str, img_len=229, split_cycle=7,
                 start_date=start_date, end_date=end_date,
                 skip=1, max_len=None, transform=None):
        """
        :param root_dir: path to the Exp0 directory
        :param img_len: the length that the images will be resized to
        :param split_cycle: amount of days the data will be split by
        :param skip: how many frames to skip between ones taken
        :param max_len: the max amount of images to use; if None - no limit
        :param transform: optional transform to be applied on each frame
        """
        if max_len is None:
            max_len = 10000

        self.root_dir = root_dir
        self.lwir_dirs = sorted(glob.glob(root_dir + '/*LWIR'))[::skip]
        self.lwir_dirs = self._filter_dirs(self.lwir_dirs, start_date, end_date)

        self.plant_crop_len = 60
        self.out_len = img_len
        self.split_cycle = split_cycle
        self.max_len = max_len

        if transform is None:
            self.transform = transforms.Compose([])
        else:
            self.transform = transform

    def _filter_dirs(self, dirs, start_date, end_date):
        format = f"{self.root_dir}/%Y_%m_%d_%H_%M_%S_LWIR"

        filtered = []
        for dir in dirs:
            date = datetime.strptime(dir, format)

            if start_date <= date <= end_date:
                filtered.append(dir)

        return filtered

    def __len__(self):
        return len(positions) * self.split_cycle

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError()

        # the day in the cycle this sample belongs to
        cycle_day = idx // len(positions)
        plant = idx % len(positions)

        to_tensor = transforms.ToTensor()

        tensors = []
        cur_day = self._get_day(self.lwir_dirs[0])

        for lwir_dir in self.lwir_dirs:
            # update the current day when it changes
            if cur_day != self._get_day(lwir_dir):
                cur_day = self._get_day(lwir_dir)
                cycle_day -= 1

            # get the image only every split_cycle days
            if not cycle_day % self.split_cycle == 0:
                continue

            try:
                image = self._get_image(lwir_dir, plant)
                tensors.append(to_tensor(image).float())
            except DirEmptyError:
                pass

        tensors = tensors[:self.max_len]
        tensors = [self.transform(tensor) for tensor in tensors]
        image = torch.cat(tensors)

        sample = {'image': image, 'label': labels[plant],
                  'position': positions[plant], 'plant': plant}

        return sample

    def _get_image(self, lwir_dir, plant_idx):
        pos = positions[plant_idx]

        left = pos[0] - self.plant_crop_len // 2
        right = pos[0] + self.plant_crop_len // 2
        top = pos[1] - self.plant_crop_len // 2
        bottom = pos[1] + self.plant_crop_len // 2

        image_path = glob.glob(lwir_dir + '/*.tiff')
        if len(image_path) == 0:
            raise DirEmptyError()

        image_path = image_path[0]

        image = Image.open(image_path)
        image = image.crop((left, top, right, bottom))
        image = image.resize((self.out_len, self.out_len))

        return image

    # returns the date (day) of the directory
    def _get_day(self, lwir_dir):
        lwir_dir = lwir_dir[len(self.root_dir)+1:]
        return lwir_dir.split('_')[2]
