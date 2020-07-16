from datetime import datetime
from typing import NamedTuple, Dict, Tuple, List, Any
import torchvision.transforms as T
from .transformations import RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, GreyscaleToRGB


class ExpPositions:
    def __init__(self, dictionary: Dict[str, Tuple[Tuple[int, int], ...]]):
        for key, value in dictionary.items():
            setattr(self, key, value)


class ExpInfo(NamedTuple):
    start_date: datetime
    end_date: datetime
    modalities_norms: Dict[str, Tuple[List[float], List[float]]]


'''

Transforms

Rescale: to scale the image
RandomCrop: to crop from image randomly. This is data augmentation.
ToTensor: to convert the numpy images to torch images (we need to swap axes).

'''



def get_experiment_modalities_params(exp_info: ExpInfo, lwir_skip: int, lwir_max_len: int, vir_max_len: int,
                                     color_max_len: int = None):
    modalities: Dict[str, Dict] = {
        'lwir': {
            'max_len': lwir_max_len, 'skip': lwir_skip, 'transform': T.Compose(
                [T.Normalize(*exp_info.modalities_norms['lwir']), T.ToPILImage(),
                 RandomCrop((229, 229)), RandomHorizontalFlip(),
                 RandomVerticalFlip(), T.ToTensor(), GreyscaleToRGB()])
        }
    }

    if 'color' in exp_info.modalities_norms.keys():
        modalities['color'] = {
            'max_len': color_max_len, 'transform': T.Compose(
                [T.Normalize(*exp_info.modalities_norms['color']), T.ToPILImage(),
                 RandomCrop((229, 229)), RandomHorizontalFlip(),
                 RandomVerticalFlip(), T.ToTensor()])
        }

    modalities.update(
        {# this is VIR, should be changed
            mod: {
                'max_len': vir_max_len, 'transform': T.Compose(
                    [T.Normalize(*norms), T.ToPILImage(),
                     RandomCrop((458, 458)), RandomHorizontalFlip(),
                     RandomVerticalFlip(), T.ToTensor(), GreyscaleToRGB()])
            } for mod, norms in exp_info.modalities_norms.items() if mod != 'lwir' and mod != 'color' # if not changed - other modalities should be added here as well
        }
    )

    return modalities


def get_all_modalities():
    return tuple(set(sum([list(info.modalities_norms.keys()) for info in experiments_info.values()], [])))


def get_experiment_modalities(exp_name: str):
    return list(experiments_info[exp_name].modalities_norms.keys())


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
    'Exp1': ExpInfo(
        datetime(2019, 7, 28),
        datetime(2019, 8, 4),
        {
            'lwir': ([21458.6621], [119.2115]),
            '577nm': ([.0046], [.0043]),
            '692nm': ([.0181], [.0178]),
            '732nm': ([.0172], [.0794]),
            'polar_a': ([2.0094], [3.3219])
        }
    ),
    'Exp2': ExpInfo(
        datetime(2019, 9, 20),
        datetime(2019, 10, 13),
        {
            'lwir': ([21150.4258], [169.5550]),
            '577nm': ([0.0341], [0.0541]),
            '692nm': ([.2081], [.2807]),
            '732nm': ([.2263], [.6049]),
            '970nm': ([.0035], [.0038]),
            'noFilter': ([.05136], [.22331]),
            'color': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        }
    ),
    'Exp3': ExpInfo(
        datetime(2019, 10, 28),
        datetime(2019, 12, 7),
        {# Normalization according to mean and StdDev
            'lwir': ([21596.1055], [139.9253]),
            '577nm': ([0.0266], [0.0574]),
            '692nm': ([.1453], [.3015]),
            '732nm': ([.1578], [.6006]),
            '970nm': ([.0026], [.0045]),
            'noFilter': ([1.4025], [3.6011]),
            'color': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        }
    ),
}

# plants are indexed left to right, top to bottom
plant_positions = {
    'Exp0': ExpPositions({
        'lwir_positions': (
            (146, 105), (206, 100), (265, 97), (322, 98), (413, 105), (464, 105), (517, 110), (576, 115),
            (149, 157), (212, 152), (262, 145), (320, 142), (416, 167), (468, 165), (522, 169), (575, 171),
            (155, 207), (213, 205), (264, 204), (322, 200), (417, 213), (467, 218), (522, 216), (573, 219),
            (157, 263), (212, 261), (267, 258), (321, 260), (418, 266), (470, 266), (528, 263), (574, 270),
            (156, 317), (212, 315), (265, 315), (327, 319), (418, 321), (468, 314), (522, 314), (574, 319),
            (154, 366), (215, 368), (269, 372), (326, 374), (417, 373), (465, 375), (520, 373), (573, 369)
        ),
        'vir_positions': (
            (1290, 670), (1730, 620), (2150, 590), (2580, 590), (3230, 630), (3615, 620), (4000, 640), (4470, 620),
            (1320, 1050), (1780, 990), (2150, 940), (2560, 910), (3270, 1070), (3660, 1060), (4045, 1080), (4450, 1080),
            (1367, 1419), (1794, 1380), (2162, 1367), (2583, 1346), (3281, 1404), (3654, 1452), (4053, 1431), (4449, 1436),
            (1389, 1823), (1793, 1803), (2195, 1767), (2580, 1776), (3294, 1805), (3680, 1802), (4086, 1778), (4457, 1803),
            (1397, 2211), (1794, 2199), (2189, 2189), (2639, 2205), (3303, 2201), (3675, 2159), (4064, 2147), (4467, 2177),
            (1386, 2582), (1821, 2588), (2219, 2597), (2642, 2607), (3303, 2588), (3665, 2615), (4062, 2574), (4463, 2547)
        )
    }),
    'Exp1': ExpPositions({
        'lwir_positions': (
            (74, 64), (147, 64), (213, 72), (282, 64), (408, 70), (478, 72), (530, 77), (592, 78),
            (50, 119), (148, 132), (203, 123), (288, 144), (410, 132), (481, 141), (541, 144), (604, 135),
            (57, 196), (131, 197), (196, 211), (287, 211), (419, 200), (491, 206), (547, 207), (609, 206),
            (44, 263), (137, 258), (203, 274), (293, 271), (425, 269), (488, 267), (554, 264), (610, 279),
            (61, 333), (128, 329), (207, 329), (287, 333), (426, 330), (551, 331), (554, 330), (602, 336),
            (44, 393), (129, 389), (206, 391), (290, 390), (420, 391), (496, 410), (551, 403), (610, 393),
            (62, 460), (132, 456), (203, 454), (275, 470), (410, 463), (482, 473), (548, 465), (610, 456)
        ),
        'vir_positions': (
            (640, 270), (1292, 260), (1705, 275), (2287, 276), (3021, 340), (3673, 306), (4081, 270), (4590, 265),
            (560, 724), (1200, 815), (1700, 828), (2275, 842), (3132, 723), (3731, 764), (4168, 818), (4650, 804),
            (533, 1312), (1053, 1231), (1625, 1425), (2250, 1300), (3152, 1300), (3730, 1265), (4230, 1310), (4780, 1255),
            (500, 1765), (1200, 1800), (1716, 1860), (2350, 1786), (3205, 1780), (3725, 1730), (4245, 1690), (4720, 1730),
            (696, 2319), (1183, 2302), (1707, 2268), (2360, 2312), (3176, 2266), (3819, 2173), (4299, 2125), (4780, 2223),
            (445, 2823), (1200, 1524), (1643, 2406), (2381, 2350), (3137, 2273), (3825, 2186), (4320, 2155), (4802, 2228),
            (669, 3342), (1237, 3287), (1769, 3326), (2334, 3338), (3180, 3294), (3762, 3269), (4272, 3249), (4855, 3186)
        )
    }),
    'Exp2': ExpPositions({
        'lwir_positions': (
            (56, 101), (113, 97), (168, 90), (233, 85), (296, 81), (395, 79), (466, 76), (527, 76), (593, 81),
            (54, 181), (118, 174), (172, 170), (233, 165), (296, 167), (398, 159), (468, 156), (536, 158), (596, 148),
            (52, 241), (110, 243), (171, 245), (225, 246), (297, 242), (398, 230), (475, 226), (539, 223), (595, 221),
            (49, 231), (102, 231), (166, 323), (232, 322), (305, 313), (410, 312), (470, 297), (539, 297), (599, 291),
            (50, 381), (106, 381), (166, 383), (238, 386), (306, 386), (414, 380), (488, 372), (542, 370), (603, 373),
            (54, 453), (111, 450), (173, 455), (233, 453), (307, 456), (416, 455), (483, 459), (538, 457), (598, 463)
        ),
        'vir_positions': (
            (581, 629), (1039, 611), (1441, 553), (1925, 503), (2378, 452), (3115, 437), (3643, 410), (4094, 398), (4616, 404),
            (586, 1235), (1089, 1172), (1482, 1149), (1927, 1109), (2386, 1102), (3137, 1038), (3656, 1000), (4192, 990), (4624, 915),
            (588, 1697), (1039, 1695), (1483, 1697), (1892, 1698), (2415, 1661), (3149, 1568), (3724, 1529), (4176, 1491), (4614, 1464),
            (586, 2279), (989, 2234), (1465, 2263), (1947, 2247), (2482, 2174), (3239, 2066), (3692, 2031), (4202, 2020), (4665, 1969),
            (626, 2713), (1024, 2690), (1468, 2713), (2014, 2705), (2515, 2697), (3299, 2655), (3838, 2589), (4227, 2559), (4685, 2580),
            (626, 3264), (1075, 3225), (1532, 3243), (1968, 3237), (2519, 3235), (3318, 3200), (3811, 3218), (4244, 3218), (4709, 3256)
        ),
        'color_positions': (
            (1004, 1091), (1195, 1085), (1409, 1065), (1651, 1041), (1892, 1032), (2255, 1020), (2521, 1005), (2758, 1006), (3039, 1000),
            (966, 1390), (1221, 1373), (1435, 1363), (1663, 1334), (1887, 1358), (2268, 1319), (2520, 1311), (2788, 1310), (3025, 1270),
            (956, 1633), (1189, 1624), (1418, 1644), (1627, 1645), (1904, 1619), (2265, 1592), (2555, 1569), (2808, 1551), (3042, 1538),
            (939, 1943), (1158, 1910), (1395, 1915), (1666, 1920), (1933, 1892), (2308, 1843), (2545, 1825), (2807, 1824), (3053, 1796),
            (957, 2153), (1156, 2148), (1397, 2154), (1676, 2147), (1934, 2148), (2343, 2147), (2635, 2124), (2841, 2110), (3064, 2115),
            (967, 2364), (1183, 2404), (1421, 2428), (1653, 2431), (1930, 2428), (2358, 2416), (2591, 2434), (2815, 2425), (3071, 2504)
        )
    }),
    'Exp3': ExpPositions({
        'lwir_positions': (
            (86, 108), (186, 100), (294, 98),  (388, 100), (484, 104), (566, 100),
            (84, 194), (188, 194), (300, 196), (398, 198), (486, 198), (572, 198),
            (86, 278), (198, 282), (302, 282), (398, 290), (488, 286), (570, 290),
            (88, 360), (198, 362), (304, 364), (404, 364), (484, 356), (576, 358),
            (88, 434), (204, 430), (316, 440), (406, 440), (488, 432), (576, 432)
        ),
        'vir_positions': (
            (837, 679) , (1619, 615), (2363, 593), (3067, 601), (3787, 601), (4399, 565), 
            (863, 1329),(1611, 1319),(2429, 1313),(3153, 1313),(3791, 1307),(4443, 1299), 
            (873, 1943),(1707, 1961),(2473, 1933),(3161, 1971),(3825, 1949),(4465, 1981), 
            (887, 2557),(1727, 2533),(2489, 2555),(3235, 2519),(3805, 2461),(4509, 2481), 
            (899, 3093),(1773, 3025),(2591, 3093),(3249, 3081),(3861, 3019),(4521, 3021)
        ),
        'color_positions': (
            (1091, 1103),(1501, 1093),(1875, 1093),(2227, 1109),(2585, 1113),(2909, 1089),
            (1107, 1459),(1489, 1451),(1909, 1447),(2255, 1465),(2587, 1457),(2915, 1467),
            (1109, 1759),(1529, 1771),(1915, 1767),(2259, 1795),(2601, 1791),(2921, 1807),
            (1107, 2059),(1533, 2061),(1917, 2077),(2289, 2073),(2581, 2039),(2945, 2067),
            (1103, 2343),(1551, 2315),(1957, 2359),(2299, 2341),(2599, 2327),(2957, 2341)
        )
    })
}


