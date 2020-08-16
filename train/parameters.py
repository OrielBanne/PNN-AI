########################################################################################################
#                                                                                                      #
#                                  PARAMETERS AND HYPER PARAMETERS                                     #
#                                                                                                      #
########################################################################################################


import numpy as np
from datasets import labels
from datasets.experiments import experiments_info
import datetime

print('~~~~~~~~~~~~~IMPORT PARAMETERS~~~~~~~~~~~~~')

# Default Parameters
experiment = 'Exp3'  # Experiment name
experiment_path = f'/home/pnn/experiments/{experiment}/'
num_days = None  # should be None
split_cycle = 1

# TODO - calculate the size in memory so that skip is automated for each
# TODO    modality and modalities are balanced
lwir_skip = 6  # once an hour picture
lwir_max_len = 150
vir_skip = 1
vir_max_len = 30
color_skip = 1
color_max_len = 30

load_checkpoint = False
use_checkpoints = False

start_date = experiments_info[experiment].start_date
# end_date = experiments_info[experiment].end_date
end_date = start_date + datetime.timedelta(hours=3)

# single plant random crop size for the different modalities:
lwir_crop = 229
color_crop = 229
depth_crop = 229
vir_crop = 458

# Choose Device
cdevice = "cuda:0"

excluded_modalities = ['noFilter', 'polar', 'polar_a', '970nm', '577nm']  # , 'depth', '732nm', '692nm', 'lwir', 'color'

excluded_plants = []
test_plants = []

batch_size = 4
loss_delta = 0.02  # The minimum above the best loss that is allowed for the model to be  saved
return_epochs = 0  # Epochs without improvement allowed. 0 to disable
epochs = 50

# For SGD Optimizer
# lamda = 0.02
# label_lr = lamda  # phenotype classifier learning rate
# plant_lr = 0.3 * lamda  # plant classifier LR used for transfer learning
# extractor_lr = lamda  # feature extractor LR
# domain_adapt_lr = 0.05  # domain adaptation learning rate

# For Adam Optimizer:
lamda = 1e-4  # previous 0.0001
label_lr_Adam = 1.4*lamda  # phenotype classifier learning rate
plant_lr_Adam = 0.6 * lamda  # plant classifier LR used for transfer learning
extractor_lr_Adam = lamda  # feature extractor LR
domain_adapt_lr = 1.8*lamda  # 0.05 domain adaptation learning rate. when set to 0 there is no domain adaptation

net_features_dim = 512

optimizer = 'Adam'  # from ['SGD', 'Adam', etc.]
# Cross Validation
n_plants = len(labels.labels[experiment])
n_classes = len(np.unique(labels.labels[experiment]))
n_splits = n_plants // n_classes

# in case random_split is used
train_ratio = n_classes / n_splits
# train_ratio = 5 / 6  # Exp3 = 5/6 # Exp0 = 6 / 8
# dataset train/test ratio - '/test' denominator should be a divisor of len(test-set)

#  Clustering Parameters
compare_clusters = True  # Compare cluster evaluation results for used modalities
num_clusters = n_classes  # The number of clusters used in the evaluations, default is Exp number of phenotypes
load_features = False  # action='store_true', default=False
cluster_tsne_results = '/home/orielban/PNN/tsne_results/'
# load features -- when true Loads the features from a file, else - computed from the extractor and saved in a csv file.

# Clustering - TSNE
plot_TSNE = True  # Save a TSNE plot for used modalities
PCA = 0  # default = 0, no PCA usage . If PCA > 0 -- features will be transformed by PCA with that number of components.


def main():
    """
    file containing all parameters and hyper parameters
    """
    print('this file containing all parameters and hyper parameters')


if __name__ == '__main__':
    main()
