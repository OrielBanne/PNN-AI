from datetime import datetime
from typing import NamedTuple, Dict, Tuple, List, Any
import torchvision.transforms as T
from .transformations import RandomCrop, RandomHorizontalFlip, RandomVerticalFlip


class ExpPositions:
    def __init__(self, dictionary: Dict[str, Tuple[Tuple[int, int], ...]]):
        for key, value in dictionary.items():
            setattr(self, key, value)


class ExpInfo(NamedTuple):
    start_date: datetime
    end_date: datetime
    modalities_norms: Dict[str, Tuple[List[float], List[float]]]


def get_experiment_modalities(exp_info: ExpInfo, lwir_skip: int, lwir_max_len: int, vir_max_len: int):
    modalities: Dict[str, Dict] = {
        'lwir': {
            'max_len': lwir_max_len, 'skip': lwir_skip, 'transform': T.Compose(
                [T.Normalize(*exp_info.modalities_norms['lwir']), T.ToPILImage(),
                 RandomCrop((229, 229)), RandomHorizontalFlip(),
                 RandomVerticalFlip(), T.ToTensor()])
        }
    }

    modalities.update(
        {
            mod: {
                'max_len': vir_max_len, 'transform': T.Compose(
                    [T.Normalize(*norms), T.ToPILImage(),
                     RandomCrop((458, 458)), RandomHorizontalFlip(),
                     RandomVerticalFlip(), T.ToTensor()])
            } for mod, norms in exp_info.modalities_norms.items() if mod != 'lwir'
        }
    )

    return modalities


experiments_info: Dict[str, ExpInfo] = {
    'Exp0': ExpInfo(
        datetime(2019, 6, 5),
        datetime(2019, 6, 19),
        {
            'lwir': ([21361.], [481.]),
            '577nm': ([.00607], [.00773]),
            '692nm': ([.02629], [.04364]),
            '732nm': ([.01072], [.11680]),
            '970nm': ([.00125], [.00095]),
            'polar': ([.05136], [.22331]),
        }
    ),
}

# plants are indexed left to right, top to bottom
plant_positions = {
    'Exp0': ExpPositions({
                'lwir_positions': (
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
                ),
                'vir_positions': (
                    (1290, 670), (1730, 620), (2150, 590), (2580, 590),
                    (3230, 630), (3615, 620), (4000, 640), (4470, 620),
                    (1320, 1050), (1780, 990), (2150, 940), (2560, 910),
                    (3270, 1070), (3660, 1060), (4045, 1080), (4450, 1080),
                    (1367, 1419), (1794, 1380), (2162, 1367), (2583, 1346),
                    (3281, 1404), (3654, 1452), (4053, 1431), (4449, 1436),
                    (1389, 1823), (1793, 1803), (2195, 1767), (2580, 1776),
                    (3294, 1805), (3680, 1802), (4086, 1778), (4457, 1803),
                    (1397, 2211), (1794, 2199), (2189, 2189), (2639, 2205),
                    (3303, 2201), (3675, 2159), (4064, 2147), (4467, 2177),
                    (1386, 2582), (1821, 2588), (2219, 2597), (2642, 2607),
                    (3303, 2588), (3665, 2615), (4062, 2574), (4463, 2547)
                )
            }),
    'Exp1': ExpPositions({
        'lwir_positions': (
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)
        ),
        'vir_positions': (
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)
        )
    }),
    'Exp2': ExpPositions({
        'lwir_positions': (
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)
        ),
        'vir_positions': (
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)
        )
    })
}
