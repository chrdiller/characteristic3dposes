import math
from typing import List

import torch
import torch.nn as nn

from characteristic3dposes.ops.loss import HeatmapOffsetGenerator
from characteristic3dposes.model.attention import AttentionModel


class PositionalEncoder(nn.Module):
    """
    Positional encoder: Encodes the joint index prior to processing all joints with an attention module
    """

    def __init__(self, d_model, seq_len=25):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(seq_len, d_model)
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]

        return x


class Characteristic3DPosesModel(nn.Module):
    def __init__(self, conf):
        """
        Build the model for characteristic 3d pose prediction, consisting of an encoder, attention module, and volumetric decoder

        :param conf: The entire configuration
        """

        super().__init__()

        # Assignments
        self.conf = conf
        self.num_total_body_joints = 17 if conf.data.type == 'h36m' else 25
        self.num_out_joints = len(conf.training.joint.predict)
        prob_out_channels = conf.training.heatmap.num_classes if conf.training.heatmap.criterion == 'ce' else 1

        # Input pose sequence encoding
        self.pose_sequence_encoder = nn.GRU(input_size=3, hidden_size=conf.model.dim_embedding, num_layers=conf.model.gru_layers)

        # Previous joint encoding
        if len(self.conf.training.joint.given) > 0:
            self.previous_joint_encoder = nn.GRU(input_size=3, hidden_size=conf.model.dim_embedding, num_layers=conf.model.gru_layers, dropout=0)

        # Attention Model with positional augmentation
        self.positional_encoder = PositionalEncoder(d_model=conf.model.dim_embedding, seq_len=self.num_total_body_joints)
        self.attention_model = AttentionModel(conf.model.dim_embedding, conf.model.p_dropout, conf.model.attention, num_out_joints=self.num_out_joints)

        # Input offsets generator
        self.input_offsets_generator = HeatmapOffsetGenerator(heatmap_resolution=16)

        # Decoder heads and tail for heatmaps and offsets; after the processing with _head, an intermediate small
        # heatmap is output which is supervised with a PoseHeatMapCriterion; the output of _head is the processed
        # with _tail which produces the final full-resolution heatmap
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.num_out_joints, out_channels=32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=self.num_out_joints * prob_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.num_out_joints * prob_out_channels),
            nn.LeakyReLU()
        )
        self.heatmap_tail = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.num_out_joints * prob_out_channels, out_channels=32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=self.num_out_joints * prob_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.num_out_joints * prob_out_channels),
            nn.Sigmoid()
         )
        self.offsets_head = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.num_out_joints, out_channels=32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
        )
        self.offsets_tail = nn.Sequential(
            nn.ConvTranspose3d(in_channels=32+self.num_total_body_joints*3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=self.num_out_joints * 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, input_skeletons: torch.Tensor, input_joint_indices: List, previous_joints: torch.Tensor):
        b = input_skeletons.shape[0]

        # Input encoding
        encoded_pose_sequence = self.pose_sequence_encoder(input_skeletons.transpose(1, 0).contiguous().view(10, -1, 3))[1][-1].view(b, self.num_total_body_joints, -1)
        if len(self.conf.training.joint.given) > 0:
            encoded_previous_joints = [self.previous_joint_encoder(previous_joint.transpose(1, 0).contiguous())[1][-1].view(b, 1, -1) for previous_joint in previous_joints]
        else:
            encoded_previous_joints = []

        # Attention
        encoded_pose_sequence = self.positional_encoder(encoded_pose_sequence)
        attention_feature = self.attention_model(encoded_pose_sequence[:, tuple(input_joint_indices), :], encoded_pose_sequence,
                                                 encoded_previous_joints,
                                                 action_feature=None, mask=None, return_attns=False)

        # Heatmap decoding
        latent_cube = attention_feature[:, :self.num_out_joints, :].view(b, self.num_out_joints, 4, 4, 4)
        heatmap_small = self.heatmap_head(latent_cube).view(b, self.num_out_joints, -1, 8, 8, 8)
        heatmap = self.heatmap_tail(heatmap_small.view(b, -1, 8, 8, 8)).view(b, self.num_out_joints, -1, 16, 16, 16)

        # Offsets decoding, taking in offsets of the last input skeleton
        input_offsets = self.input_offsets_generator(input_skeletons[:, -1]).permute(0, 1, 5, 2, 3, 4).contiguous().view(b, -1, 16, 16, 16)
        offsets = self.offsets_tail(torch.cat([self.offsets_head(latent_cube), input_offsets], dim=1)).view(b, self.num_out_joints, -1, 16, 16, 16)

        return {
            'heatmap': heatmap,
            'heatmap_small': heatmap_small,
            'offsets': offsets
        }
