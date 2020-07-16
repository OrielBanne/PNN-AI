########################################################################################################
#                                                                                                      #
#                                         FeatureExtraction                                            #
#                                                                                                      #
########################################################################################################


import torch
from torch import nn
from torchvision.models import inception_v3
# from TCN.tcn import TemporalConvNet
from model.TCN import TemporalConvNet
from typing import Dict, Union
from datetime import timedelta, datetime, time
import yaml


def greyscale_to_RGB(image: torch.Tensor, add_channels_dim=False) -> torch.Tensor:
    if add_channels_dim:
        image = image.unsqueeze(-3)

    dims = [-1] * len(image.shape)
    dims[-3] = 3
    return image.expand(*dims)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x

    # INCEPTION V3


class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        ## Note - inception_V3 expects tensors with a size of Nx3x299x299, ensure this size!
        #  So why are we using 229 on 229 images?
        self.inception = inception_v3(pretrained=True, transform_input=False, aux_logits=True)
        ## pretrained      - if true, returns a model pre-trained on ImageNet.
        ## progress        - if true, displays a progress bar of the download to stderr ## should try this
        ## aux_logits      - if true, add an auxilary branch that can improve training. default =True
        ## transform_input - if True, preprocesses the input according to the method with which it was trained on imageNet. default = False

        self.inception.fc = Identity()  # adding a last layer FC layer identity
        self.inception.eval()  # evaluation mode - during testing we do not want dropout to take place - correct???

        for p in self.inception.parameters():
            p.requires_grad = False  # here we freeze the pretrained inception model parameters

    # make sure that the inception model stays on eval
    def train(self, mode=True):
        return self

    def forward(self, x: torch.Tensor):
        """
        :param x: a batch of image sequences of either size (NxTxCxHxW) or the squeezed size (NxTxHxW)
        :return: a batch of feature vectors for each image of size (NxTx2048)
        """
        # if the image is greyscale convert it to RGB
        if (len(x.shape) < 5) or (len(x.shape) >= 5 and x.shape[-3] == 1):  # ??????
            x = greyscale_to_RGB(x, add_channels_dim=len(x.shape) < 5)

        # if we got a batch of sequences we have to calculate each sequence separately
        N, T = x.shape[:2]
        return self.inception(x.view(-1, *x.shape[2:])).view(N, T, -1)

    # Temporal Convolutional Net


class ModalityFeatureExtractor(nn.Module):
    def __init__(self, num_levels: int = 3, num_hidden: int = 64, embedding_size: int = 128, kernel_size=2,
                 dropout=0.2):
        """
        :param num_levels: the number of TCN layers
        :param num_hidden: number of channels used in the hidden layers
        :param embedding_size: size of final feature vector
        :param kernel_size: kernel size, make sure that it matches the feature vector size
        :param dropout: dropout probability
        :return: a TemporalConvNet matching the inputted params
        """
        super().__init__()
        num_channels = [num_hidden] * (num_levels - 1) + [embedding_size]
        self.tcn = TemporalConvNet(2048, num_channels, kernel_size, dropout)

    def forward(self, x: torch.Tensor):
        """
        :param x: input tensor of size (NxTx2048),
                where N is the batch size, T is the sequence length and 2048 is the input embedding dim
        :return: tensor of size (N x T x embedding_size) where out[:,t,:] is the output given all values up to time t
        """

        # transpose each sequence so that we get the correct size for the TemporalConvNet
        x = torch.stack([m.t() for m in x])

        # Forward through TCN:
        out = self.tcn(x)

        # undo the previous transpose
        return torch.stack([m.t() for m in out])


'''
in plant feature extractor def - param_mods looks like so:

feat_extractor_params dictionary :
577nm:
  kernel_size: 2
  num_levels: 2
  etc ....
'''


class PlantFeatureExtractor(nn.Module):
    def __init__(self, *default_mods: str, embedding_size: int = 512, **param_mods: Dict[str, Union[int, float]]):
        super().__init__()

        # make sure modality does not appear as both default and with params
        assert len(set(default_mods).intersection(param_mods.keys())) == 0

        # all modalities
        self.mods = list(default_mods) + list(param_mods.keys())

        # make sure that we are using ANY modalities
        assert len(self.mods) > 0

        # create a feature extractor for images - Inception
        self.image_feat_ext = ImageFeatureExtractor()

        # a dictionary for the feature extractors for each modality
        self.mod_extractors: Dict[str, nn.Module] = dict()

        # create a feature extractor with the inputted params for each param modality - TCN
        for mod in param_mods.keys():
            self.mod_extractors[mod] = ModalityFeatureExtractor(**param_mods[mod])
            self.add_module(f'TCN_{mod}_feat_extractor', self.mod_extractors[mod])

        # final feature extractor - linear FC
        D_in = 128 * len(self.mods)  # Exp0: 128* 5 mods = 640//Exp2: 128*6 mods = 768
        D_out = embedding_size  # 512
        H = int((D_in + D_out) / 2)  # 640

        print('D_in  = ', D_in, 'D_out  = ', D_out, ' H = ', H)
        self.final_feat_extractor = nn.Linear(D_in, D_out)  # previous layer

        # This is yet without convolution, but has 2 layers and ReLU activation - might improve
        # self.final_feat_extractor = torch.nn.Sequential(
        #    torch.nn.Linear(D_in, H),
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(H, D_out),
        # )

        # Expected 3-dimensional input for 3-dimensional weight [640, 768, 1], but got 2-dimensional input
        # of size [4, 768] instead
        # D_in  =  768 D_out  =  512  H =  640  batch_size = 4
        # input size was [4, 768]  = [batch_size, D_in] # I checked it is batch size
        # Convolution added checking dimensions
        # self.final_feat_extractor = torch.nn.Sequential(
        #    torch.nn.Conv1d(D_in, H, kernel_size = 1, stride=1, padding=0, dilation=1, groups=1, bias=True,
        #    padding_mode='zeros'),
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(H, D_out),
        # )
        '''
        torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        
        N is a batch size, C denotes a number of channels, L is a length of signal sequence.
        layer with input size (N,Cin,L) and output (N,Cout,Lout) 
        
        batch len is batch_size:  8
        x len is 6 as we have erased   
            del x['label']
            del x['plant']
        so there are 6 modalities per x, in a dictionary structure
        
        
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding) – applies convolution
        torch.nn.relu(x) – applies ReLU
        torch.nn.MaxPool2d(kernel_size, stride, padding) – applies max pooling
        torch.nn.Linear(in_features, out_features) – fully connected layer (multiply inputs by learned weights)

        final layer D_in = 

        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
        )
        
        '''

        self.device = None
        self.streams = {mod: None for mod in self.mods}

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

        # streams - makes Cuda work in parallel on 2 GPU's (check)
        if next(self.parameters()).is_cuda:
            device = next(self.parameters()).get_device()
            if self.device != device:
                self.device = device
                self.streams = {mod: torch.cuda.Stream(device=self.device) for mod in self.mods}
        elif self.device is not None:
            self.device = None
            self.streams = {mod: None for mod in self.mods}

        return self

    def forward(self, **x: torch.Tensor):
        # print(' @@@@ forward in PlantFeatureExtractor class             @@@@@  ', datetime.now())
        """

        :param x: input of type mod=x_mod for each modality where each x_mod is of shape
                    (NxT_modxC_modxH_modxW_mod),
                    where the batch size N is the same over all mods and others are mod-specific
        :return: a feature vector of shape (Nxembedding_size)
        """
        # make sure that all of the extractor mods and only them are used
        assert set(self.mods) == set(x.keys())

        # extract features from each image
        if self.device is None:
            img_feats = {mod: self.image_feat_ext(x[mod]) for mod in self.mods}
        else:
            img_feats = {}
            for mod in self.mods:
                with torch.cuda.stream(self.streams[mod]):
                    img_feats[mod] = self.image_feat_ext(x[mod])

        # extract the features for each mod using the corresponding feature extractor
        if self.device is None:
            mod_feats = {mod: self.mod_extractors[mod](img_feats[mod]) for mod in self.mods}
        else:
            mod_feats = {}
            for mod in self.mods:
                with torch.cuda.stream(self.streams[mod]):
                    mod_feats[mod] = self.mod_extractors[mod](img_feats[mod])  ######
            for mod in self.mods:
                self.streams[mod].synchronize()

        # take the final feature vector from each sequence
        combined_features = torch.cat([mod_feats[mod][:, -1, :] for mod in self.mods], dim=1)

        return self.final_feat_extractor(combined_features)
