import glob

'''
The glob module finds all the path-names matching a specified pattern according to the rules used by the Unix shell, 
although results are returned in arbitrary order. No tilde expansion is done, but *, ?, and character ranges expressed 
with [] will be correctly matched. This is done by using the os.listdir() and fnmatch.fnmatch() functions in concert, 
and not by actually invoking a sub-shell. Note that unlike fnmatch.fnmatch(), glob treats file-names beginning with a 
dot (.) as special cases. (For tilde and shell variable expansion, use os.path.expanduser() and os.path.expandvars().)


For a literal match, wrap the meta-characters in brackets. For example, '[?]' matches the character '?'.

'''
from PIL import Image
from torchvision.transforms import ToTensor

from .experiments import plant_positions
from .ModalityDataset import ModalityDataset, DirEmptyError
from train import parameters


class Color(ModalityDataset):
    def __init__(self, root_dir: str, exp_name: str, img_len: int = 255, split_cycle=parameters.split_cycle,
                 start_date=parameters.start_date, end_date=parameters.end_date, skip=parameters.color_skip,
                 max_len=None, transform=None):
        super().__init__(root_dir, exp_name, 'D465_Color', img_len, plant_positions[exp_name].color_positions,
                         split_cycle, start_date, end_date, skip=1, max_len=parameters.color_max_len,
                         transform=None)

    def _get_image(self, directory, plant_position):
        left = plant_position[0] - self.img_len // 2
        right = plant_position[0] + self.img_len // 2
        top = plant_position[1] - self.img_len // 2
        bottom = plant_position[1] + self.img_len // 2

        image_path = glob.glob(directory + '/*.jpg')
        if len(image_path) == 0:
            raise DirEmptyError()

        image_path = image_path[0]

        image = Image.open(image_path)
        image = image.crop((left, top, right, bottom))

        to_tensor = ToTensor()

        return to_tensor(image).float()

    def _dir_has_file(self, directory) -> bool:
        return len(glob.glob(directory + '/*.jpg')) != 0
