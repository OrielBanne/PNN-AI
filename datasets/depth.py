""" Depth Image data handling"""
import glob
from PIL import Image
from torchvision.transforms import ToTensor
from datasets.experiments import plant_positions
from datasets.ModalityDataset import ModalityDataset, DirEmptyError
from train import parameters
import numpy as np
from matplotlib import cm  # Color Map


class Depth_day_night(ModalityDataset):
    """
    The Depth data from the experiment.
    """

    def __init__(self, root_dir: str, exp_name: str, img_len=255, split_cycle=parameters.split_cycle,
                 plant_crop_len: int = 65, start_date=parameters.start_date,
                 end_date=parameters.end_date, skip=1, max_len=None, transform=None):
        """
        :param plant_crop_len: the size of the area around each plant that will be taken as an image
        """
        super().__init__(root_dir, exp_name, 'Depth_day_night', img_len, plant_positions[exp_name].depth_positions,
                         split_cycle, start_date, end_date, skip, max_len, transform)
        self.plant_crop_len = plant_crop_len
        self.img_len = img_len

    @staticmethod
    def read_depth_image(image_path, bit_type, w, h):
        raw_image = np.fromfile(image_path, dtype=bit_type)
        # TODO: Clip image to 34000<value<38000 and normalize the data
        return raw_image[:h * w].reshape(h, w)

    def _get_image(self, directory, plant_position):
        left = plant_position[0] - self.img_len // 2
        right = plant_position[0] + self.img_len // 2
        top = plant_position[1] - self.img_len // 2
        bottom = plant_position[1] + self.img_len // 2

        image_path = glob.glob(directory + '/*ZY_Z16*.raw')
        if len(image_path) == 0:
            raise DirEmptyError()

        image_path = image_path[0]
        img = self.read_depth_image(image_path, 'uint8', 1280, 1024)  # Image.open(image_path)
        image = Image.fromarray(np.uint8(cm.gist_earth(img) * 255))
        image = image.crop((left, top, right, bottom))
        image = image.resize((self.img_len, self.img_len), resample=Image.NEAREST)

        to_tensor = ToTensor()

        return to_tensor(image).float()

    def _dir_has_file(self, directory) -> bool:
        return len(glob.glob(directory + '/*ZY_Z16*.raw')) != 0
