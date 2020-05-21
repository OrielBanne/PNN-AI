import argparse


def add_experiment_dataset_arguments(parser: argparse.ArgumentParser):
    split_cycle=7 #"The number of samples that each plant in the dataset will be split into.")
    lwir_max_len=44 # default=None The maximum number of images in a single lwir sample. 
                    #If not used it is unlimited, and if used with no number (i.e using - 
                    # -lwir_max_len with no value) it will have a default of 44
    vir_max_len=6   # default=None, The maximum number of images in a single vir sample.
                    # If not used it is unlimited, and if used with no number (i.e using --vir_max_len with no value)-->6
    color_max_len=6 # default=None The maximum number of images in a single color sample.
                    # If not used it is unlimited, and if used with no number (i.e using --color_max_len with no value)-->6
    lwir_skip=5     # default=1  max num of images for single vir sample. when not used = 1, used with no number (i.e using --lwir_skip or --skip with                      # no value) -->5
    num_days=None   # default is all of days from start
    experiment = Exp0# required=True, choices=['Exp0', 'Exp1', 'Exp2'],
    experiment_path= None # default=None, Path to the experiment root directory


def __get_name(exp_name: str, file_type: str, excluded_modalities=[]):
    if len(excluded_modalities) == 0:
        return f'{exp_name}_{file_type}_all'
    else:
        excluded_modalities.sort()
        return '_'.join([exp_name, file_type, 'no'] + excluded_modalities)


def get_checkpoint_name(exp_name: str, excluded_modalities=[]):
    return __get_name(exp_name, 'checkpoint', excluded_modalities)


def get_feature_file_name(exp_name: str, excluded_modalities=[]):
    return f"{__get_name(exp_name, 'features', excluded_modalities)}.csv"


def get_tsne_name(exp_name: str, excluded_modalities=[], pca=0):
    return __get_name(exp_name, f'TSNE_{pca}' if pca > 0 else 'TSNE', excluded_modalities)


def get_used_modalities(modalities, excluded_modalities=[]):
    return {mod: arguments for mod, arguments in modalities.items() if mod not in excluded_modalities}


def get_levels_kernel(history_len: int):
    if history_len <= 7:
        kernel_size = 2
        num_levels = 2
    elif 7 <= history_len <= 15:
        kernel_size = 2
        num_levels = 3
    elif 15 <= history_len <= 25:
        kernel_size = 5
        num_levels = 2
    elif 25 <= history_len <= 57:
        kernel_size = 5
        num_levels = 3
    elif 57 <= history_len <= 91:
        kernel_size = 4
        num_levels = 4
    elif 91 <= history_len <= 121:
        kernel_size = 5
        num_levels = 4
    elif 121 <= history_len <= 249:
        kernel_size = 5
        num_levels = 5
    elif 249 <= history_len <= 311:
        kernel_size = 6
        num_levels = 5
    else:
        kernel_size = 5
        num_levels = 6

    return num_levels, kernel_size
