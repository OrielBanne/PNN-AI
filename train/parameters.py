########################################################################################################
#                                                                                                      #
#                                  PARAMETERS AND HYPER PARAMETERS                                     #
#                                                                                                      #
########################################################################################################

from datetime import datetime, timedelta

print('~~~~~~~~~~~~~IMPORT PARAMETERS~~~~~~~~~~~~~')


# Default Parameters
use_checkpoints=False #  checkpoints disable for training
load_checkpoint=False #  load previous training checkpoint 
experiment = 'Exp0'   #  Experiment name 
experiment_path = f'/home/orielban/experiments/{experiment}/'
load_features = True
num_days = None      ### should be None
split_cycle = 7
start_date = None
lwir_max_len = 64
lwir_skip = 6
vir_max_len = 16     # need to understand how many of these are the max that exist
color_max_len = 16   #        "
num_days = None
load_checkpoint = False
use_checkpoints = True

# define start and end dates of the experiment here, this should be a dictionary according to the experiment

# Exp0
start_date = datetime(2019, 6, 4)
end_date   = datetime(2019, 7, 7)

# Exp1
# start_date = datetime(2019, 7, 28)
# end_date   = datetime(2019, 8, 4)

# Exp2
# start_date = datetime(2019, 9, 20)
# end_date   = datetime(2019, 10, 13)

# Exp3


# Exp3_5

# Exp3_6


# Exp4


# Choose Device
cdevice = "cuda:0"

# training hyper-parameters       #     added again , '577nm'
excluded_modalities=['noFilter', 'polar', 'polar_a'] # All of the modalities that you don't want to use
train_ratio = 5/6     #  dataset train/test ratio
batch_size=4          #  training batch size
loss_delta=0.0        #  The minimum above the best loss that is allowed for the model to be  saved
return_epochs=0       #  Epochs without improvement allowed (loss_delta included). Return to the best checkpoint otherwise. 0 to disable
epochs = 50           #  number of epochs during train
label_lr=1e-2         #  phenotype classifier learning rate
plant_lr=1e-2         #  plant classifier LR used for transfer learning
extractor_lr=1e-2     #  feature extractor LR
domain_adapt_lr=1e-2  #  domain adaptation learning rate

#  Clustering Parameters
compare_clusters = True # Compare cluster evaluation results for used modalities
num_clusters     = 6    # The number of clusters used in the evaluations, default is Exp number of phenotypes
load_features    = False# action='store_true', default=False 
cluster_tsne_results = '/home/orielban/PNN-AI/PNN-AI/tsne_results/' 
# load features -- when true Loads the features from a file, else - computed from the extractor and saved in a csv file.

# Clustering - TSNE
plot_TSNE = True # Save a TSNE plot for used modalities
PCA = 0  # default = 0, no PCA usage . If PCA > 0 -- features will be transformed by PCA with that number of components.
