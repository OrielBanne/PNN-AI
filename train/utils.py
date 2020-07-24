########################################################################################################
#                                                                                                      #
#                                               Utils                                                  #
#                                                                                                      #
########################################################################################################


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


def get_training_name(exp_name: str, excluded_modalities=[]):
    return __get_name(exp_name, '_training_', excluded_modalities)


def get_used_modalities(modalities, excluded_modalities=[]):
    return {mod: args for mod, args in modalities.items() if mod not in excluded_modalities}


def get_levels_kernel(history_len: int):
    # Effective history formula: 1 + 2*(kernel_size-1)*(2^num_levels-1)
    # history_len == Effective history
    if history_len <= 7:
        kernel_size, num_levels = 2, 2
    elif 7 <= history_len <= 15:
        kernel_size, num_levels = 2, 3
    elif 15 <= history_len <= 25:
        kernel_size, num_levels = 5, 2
    elif 25 <= history_len <= 57:
        kernel_size, num_levels = 5, 3
    elif 57 <= history_len <= 91:
        kernel_size, num_levels = 4, 4
    elif 91 <= history_len <= 121:
        kernel_size, num_levels = 5, 4
    elif 121 <= history_len <= 249:
        kernel_size, num_levels = 5, 5
    elif 249 <= history_len <= 311:
        kernel_size, num_levels = 6, 5
    else:
        # effective history: 505
        kernel_size, num_levels = 5, 6

    return num_levels, kernel_size
