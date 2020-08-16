""" LWIR Image data handling"""
import glob
from PIL import Image
from torchvision.transforms import ToTensor
from datasets.experiments import plant_positions
from datasets.ModalityDataset import ModalityDataset, DirEmptyError
from train import parameters


class LWIR(ModalityDataset):
    """
    The LWIR data from the experiment.
    """

    def __init__(self, root_dir: str, exp_name: str, img_len=255, split_cycle=parameters.split_cycle,
                 plant_crop_len: int = 65, start_date=parameters.start_date,
                 end_date=parameters.end_date, skip=1, max_len=None, transform=None):
        """
        :param plant_crop_len: the size of the area around each plant that will be taken as an image
        """
        super().__init__(root_dir, exp_name, 'LWIR', img_len, plant_positions[exp_name].lwir_positions, split_cycle,
                         start_date, end_date, skip, max_len, transform)
        self.plant_crop_len = plant_crop_len

    def _get_image(self, directory, plant_position):
        left = plant_position[0] - self.plant_crop_len // 2
        right = plant_position[0] + self.plant_crop_len // 2
        top = plant_position[1] - self.plant_crop_len // 2
        bottom = plant_position[1] + self.plant_crop_len // 2

        image_path = glob.glob(directory + '/*.tiff')
        if len(image_path) == 0:
            raise DirEmptyError()

        image_path = image_path[0]
        image = Image.open(image_path)
        image = image.crop((left, top, right, bottom))
        # image = image.resize((self.img_len, self.img_len), resample=Image.NEAREST)
        image = image.resize((self.img_len, self.img_len), resample=Image.BICUBIC)  # TODO: already changed  -check

        to_tensor = ToTensor()

        return to_tensor(image).float()

    def _dir_has_file(self, directory) -> bool:
        return len(glob.glob(directory + '/*.tiff')) != 0
