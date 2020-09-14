import torch
from torch import nn
from model.TCN import TemporalConvNet
from typing import Dict, Union
from train import parameters


class TCNextractor(nn.Module):  # Temporal Convolutional Net
    def __init__(self, num_levels: int = 3,
                 num_hidden: int = parameters.tcn_num_hidden_parameters,
                 embedding_size: int = parameters.tcn_embedding_size,
                 kernel_size=parameters.tcn_kernel_size,
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

        # print(x.shape) --> torch.Size([4, 57, 1, 2048])
        #
        x = torch.stack([m.t() for m in x])

        # Forward through TCN:
        out = self.tcn(x)

        # undo the previous transpose
        return torch.stack([m.t() for m in out])


class PlantFeatureExtractor(nn.Module):
    def __init__(self,
                 embedding_size: int = parameters.net_features_dim,
                 **mods: Dict[str, Union[int, float]]):
        super().__init__()

        # all modalities
        self.mods = list(mods.keys())

        # image features were already extracted by inception - read features with DataLoader

        # a dictionary for the feature extractors for each modality
        self.mod_extractors: Dict[str, nn.Module] = dict()

        # create a feature extractor with the inputted params for each param modality - TCN
        for mod in mods.keys():
            self.mod_extractors[mod] = TCNextractor(**mods[mod])
            self.add_module(f'TCN_{mod}_feat_extractor', self.mod_extractors[mod])

        # final feature extractor - linear FC
        d_in = parameters.tcn_embedding_size * len(self.mods)
        # Exp0: tcn_embedding_size(=128)* 5 mods = 640//
        # Exp2: tcn_embedding_size(=128)*6 mods = 768
        d_out = embedding_size  # 512
        # h = int((d_in + d_out) / 2)  # 640

        # print('D_in  = ', d_in, 'D_out  = ', d_out, ' H = ', h)
        self.final_feat_extractor = nn.Linear(d_in, d_out)  # FC Linear last layer

    def forward(self, **x: torch.Tensor):
        """
        :param x: input of type mod=x_mod for each modality where each x_mod is of shape
                    (NxT_modx2048),
                    where the batch size N is the same over all mods and others are mod-specific
        :return: a feature vector of shape (N x embedding_size)
        """

        # extract the features for each mod using the corresponding feature extractor
        mod_feats = {}
        for mod in self.mods:
            # print(mod, 'x[mod].shape ', x[mod].shape) # torch.Size([4, 57, 1, 2048])
            x[mod] = torch.squeeze(x[mod])
            # print('x[mod].shape after torch.squeeze(x[mod]) ', x[mod].shape)  #torch.Size([4, 57, 2048])

            # FORWARD:
            mod_feats[mod] = self.mod_extractors[mod](x[mod])
            # print('mod_feats[mod] shape : ', mod_feats[mod].shape)  # torch.Size([4, 57, 128])
            # print('[mod_feats[mod][:, -1, :] shape:  ', mod_feats[mod][:, -1, :].shape) # torch.Size([2, 128])
            '''   Exp0:
            732nm   torch.Size([4, 57, 1, 2048])
            692nm   torch.Size([4, 57, 1, 2048])
            lwir    torch.Size([4, 1987, 1, 2048])
            970nm   torch.Size([4, 57, 1, 2048])
            577nm   torch.Size([4, 57, 1, 2048])
            '''

        # take the final feature vector from each sequence
        combined_features = torch.cat([mod_feats[mod][:, -1, :] for mod in self.mods], dim=1)

        return self.final_feat_extractor(combined_features)
