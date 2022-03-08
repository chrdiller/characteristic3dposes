import math
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional


def mpjpe(prediction: np.array, target: np.array):
    """
    Calculate the mpjpe (l2) between 3D prediction and target vectors with numpy

    :param prediction: Predicted samples, as num_samples x 3
    :param target: Target samples, as num_samples x 3
    :return: The mpjpe per prediction - target pair (num_samples)
    """

    num_samples, num_dims = target.shape
    return np.linalg.norm(prediction.reshape(-1, 3) - target.reshape(-1, 3), ord=2, axis=1).reshape(-1, num_samples)


class MeanPerJointPositionError(torch.nn.Module):
    """
    Mean-Per-Joint-Position-Error (per-joint l2) between a predicted skeleton and a target skeleton, with pytorch
    """

    def __init__(self, keep_joints=False):
        super().__init__()
        self.keep_joints = keep_joints

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        b, num_nodes, num_dims = target.shape

        per_joint_error = torch.norm(prediction.view(-1, 3) - target.view(-1, 3), p=2, dim=1).view(-1, num_nodes)
        return per_joint_error if self.keep_joints else torch.mean(per_joint_error)


class Gaussian(torch.nn.Module):
    """
    Applying a Gaussian kernel to an n-dimensional tensor
    """

    def __init__(self, sigma: int, num_dims=1):
        super().__init__()
        kernel_size = max(2 * sigma + 1, 3)

        self.kernel_size = kernel_size
        kernel_size = [kernel_size] * num_dims
        sigma = [sigma] * num_dims
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])

        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        padding = int((self.kernel_size - 1) / 2)
        x = functional.pad(x, [padding] * 6, mode='constant')
        x = functional.conv3d(x, self.kernel)

        return x


class PoseHeatMapCriterion(torch.nn.Module):
    """
    Criterion to compare the predicted per-joint heatmaps with the targets
    Underlying criterion can be either l1, l2, smooth l1, or per-voxel ce (the default)
    """

    def __init__(self, conf: SimpleNamespace, small=False):
        """
        Initialize the PoseHeatMapCriterion

        :param conf: The full training configuration
        :param small: Whether this criterion is used for the full heatmap (16^3) or for the intermediate small version (8^3)
        """

        super(PoseHeatMapCriterion, self).__init__()
        self.gaussian = Gaussian(sigma=conf.training.heatmap.sigma if not small else conf.training.heatmap.sigma // 2, num_dims=3)
        self.heatmap_resolution = 16 if not small else 8

        if conf.training.heatmap.criterion == 'l2':
            self.criterion = torch.nn.MSELoss(reduction='none')
        elif conf.training.heatmap.criterion == 'l1':
            self.criterion = torch.nn.L1Loss(reduction='none')
        elif conf.training.heatmap.criterion == 'smooth_l1':
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
        elif conf.training.heatmap.criterion == 'ce':
            self.num_classes = conf.training.heatmap.num_classes
            test_volume = torch.zeros([1, 1, 3, 3, 3])
            test_volume[0, 0, 1, 1, 1] = 1
            test_volume = self.gaussian(test_volume)
            test_volume = ((test_volume / test_volume.max()) * float(self.num_classes)).floor()
            weight = test_volume.histc(bins=conf.training.heatmap.num_classes)
            weight[0] += (self.heatmap_resolution ** 3) - (self.gaussian.kernel_size ** 3)
            for class_idx in range(conf.training.heatmap.num_classes):
                if weight[class_idx] > 0:
                    weight[class_idx] = 1 / np.log(1.2 + weight[class_idx])

            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')
        else:
            raise ValueError

    def forward(self, predicted_volume: torch.Tensor, target_skeleton: torch.Tensor, volume_center=None, return_target_volume=False):
        b = target_skeleton.shape[0]
        target_volume = torch.zeros(predicted_volume.shape[0], 1, predicted_volume.shape[2], predicted_volume.shape[3],
                                    predicted_volume.shape[4], device=target_skeleton.device)

        # Generate indices, set a 1 there, and apply a Gaussian blur
        indices = torch.floor((((target_skeleton - (volume_center if volume_center is not None else 0.)) + 1.) * self.heatmap_resolution / 2)).clamp(0, (self.heatmap_resolution-1)).type(torch.long)
        indices = torch.cat([torch.stack([torch.arange(b, device=target_skeleton.device), torch.zeros(size=[b], device=target_skeleton.device, dtype=torch.long)], dim=0), indices.view(-1, 3).T])
        target_volume[tuple(indices)] = 1
        target_volume = self.gaussian(target_volume.view(b, 1, self.heatmap_resolution, self.heatmap_resolution, self.heatmap_resolution)).view(b, 1, self.heatmap_resolution, self.heatmap_resolution, self.heatmap_resolution)
        target_volume /= target_volume.max()

        # Compare using the underlying criterion
        if type(self.criterion) == torch.nn.CrossEntropyLoss:
            target_volume = torch.clamp(torch.floor(target_volume * float(self.num_classes)).type(torch.long), 0, self.num_classes - 1)

            loss = self.criterion(predicted_volume.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes), target_volume.view(-1))
            if return_target_volume:
                return torch.mean(loss), target_volume
            else:
                return torch.mean(loss)
        else:
            per_voxel_loss = self.criterion(predicted_volume, target_volume)
            per_voxel_loss[target_volume > 0] = per_voxel_loss[target_volume > 0]
            if return_target_volume:
                return torch.mean(per_voxel_loss), target_volume
            else:
                return torch.mean(per_voxel_loss)


class HeatmapOffsetGenerator(torch.nn.Module):
    """
    Generate per-joint volumes where each voxel contains an offset to its skeleton joint
    Used to generate offset volumes for the input (and target); the model then learn to modify it for the final skeleton prediction
    """

    def __init__(self, heatmap_resolution: int):
        super().__init__()
        self.heatmap_resolution = heatmap_resolution

    def forward(self, input_skeleton: torch.Tensor, volume_center: torch.Tensor = None):
        b, num_joints, dims = input_skeleton.shape

        output = torch.ones(size=[b, num_joints, self.heatmap_resolution, self.heatmap_resolution, self.heatmap_resolution, 3], device=input_skeleton.device)
        for joint_idx in range(num_joints):
            all = torch.ones(size=[b, self.heatmap_resolution, self.heatmap_resolution, self.heatmap_resolution], dtype=bool, device=input_skeleton.device)
            voxel_locations = torch.nonzero(all, as_tuple=False)
            batches = voxel_locations[:, 0]

            if volume_center is None:
                target_locations = input_skeleton[batches, joint_idx]
            else:
                target_locations = (input_skeleton - volume_center)[batches, joint_idx]

            output[:, joint_idx] = (target_locations - (voxel_locations[:, 1:] / (self.heatmap_resolution / 2.) - 1.)).view(b, self.heatmap_resolution, self.heatmap_resolution, self.heatmap_resolution, 3)
            output[:, joint_idx] = torch.clamp(output[:, joint_idx], -(1. / (self.heatmap_resolution / 2.)), 1. / (self.heatmap_resolution / 2.))

        return output


class HeatmapOffsetCriterion(torch.nn.Module):
    """
    Criterion to compare the predicted per-joint offset volumes with the target
    Underlying criterion can be either l1 (the default), l2, smooth l1
    """

    def __init__(self, conf: SimpleNamespace):
        """
        Initialize the HeatmapOffsetCriterion

        :param conf: The full training configuration
        """

        super().__init__()
        self.heatmap_resolution = 16
        if conf.training.offsets.criterion == 'l2':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        elif conf.training.offsets.criterion == 'l1':
            self.criterion = torch.nn.L1Loss(reduction='mean')
        elif conf.training.offsets.criterion == 'smooth_l1':
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
        self.overlap_threshold = conf.training.offsets.overlap_threshold

    def forward(self, offsets, target_skeleton, predicted_heatmap, target_heatmap, volume_center=None):
        # Only learn offsets in areas where predicted and target heatmaps overlap
        overlap = torch.logical_and(predicted_heatmap.argmax(1) >= self.overlap_threshold, target_heatmap.squeeze(1) >= self.overlap_threshold)
        offset_vectors = offsets.permute(0, 2, 3, 4, 1)[overlap]
        offset_voxel_locations = torch.nonzero(overlap, as_tuple=False)
        batches = offset_voxel_locations[:, 0]

        if volume_center is None:
            target_locations = target_skeleton[batches].squeeze(1)
        else:
            target_locations = (target_skeleton - volume_center)[batches].squeeze(1)

        target_offsets = target_locations - (offset_voxel_locations[:, 1:] / (self.heatmap_resolution / 2.) - 1.)
        target_offsets = torch.clamp(target_offsets, -(1. / (self.heatmap_resolution / 2.)), 1. / (self.heatmap_resolution / 2.))

        # Apply underlying criterion
        loss = self.criterion(offset_vectors, target_offsets) if len(offset_vectors) > 0 else torch.tensor(0., device=offsets.device, requires_grad=False)

        return loss
