from pathlib import Path

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from characteristic3dposes.data import grab
from characteristic3dposes.data.grab.constants import PHASE_SUBJECTS


def make_one_hot(idx: int, size: int) -> np.array:
    """
    Creates a 1D vector filled with zeros with given size and a one at given idx

    :param idx: Where to place the one
    :param size: 1D size of the vector
    :return: 1D vector filled with zeros with given size and a one at given idx
    """

    vec = np.zeros(shape=[size])
    vec[idx] = 1
    return vec


class GRABCharacteristicPoseDataset(torch.utils.data.Dataset):
    def __init__(self, phase, conf, input_start=None):
        """
        Characteristic Pose dataset, built on top of GRAB

        :param phase: The experiment phase: train, val, test. Affects: sample ids, augmentation, input starting time
        :param conf: The configuration for "data" (see config.yaml)
        :param input_start: The start of the input sequence, in text: ['contact', 'charpose', 'random', 'middle', 'onethird', 'twothirds']
        """

        super().__init__()

        # Assignments
        self.conf = conf
        self.phase = phase
        self.input_start = input_start if input_start is not None else conf.input_start

        # Assertions
        assert Path(conf.file).is_file()
        assert conf.type == 'grab'
        assert self.phase in ['train', 'val', 'test']
        assert self.input_start in ['contact', 'charpose', 'random', 'middle', 'onethird', 'twothirds']

        # Select sample IDs
        self.sample_ids = grab.sample_ids_for_subjects(PHASE_SUBJECTS[phase])

        # Loading actual data
        self.pose_sequences = np.load(self.conf.file, allow_pickle=True)['pose_sequences'].item()
        self.charpose_indices = np.load(self.conf.file, allow_pickle=True)['charpose_indices'].item()

        print(f"[GRABCharacteristicPoseDataset] Loaded {len(self.sample_ids)} unique sample IDs from dataset, will multiply by {self.conf.multiplicator}")
        self.sample_ids *= self.conf.multiplicator

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        # Get sample id
        sample_id = self.sample_ids[index]
        subject, action_id = grab.split_sample_id(sample_id)

        # Load data for this sample id
        pose_sequence = self.pose_sequences[subject][action_id]
        charpose_indices = self.charpose_indices[subject][action_id]

        # Frame definitions
        num_input_frames = 10
        frames_per_pose_input = 120 // 30
        contact_frame_idx = charpose_indices[1]
        charpose_frame_idx = charpose_indices[2]

        # Input sequence frame
        num_possible_shifts = (charpose_frame_idx - contact_frame_idx) // frames_per_pose_input - num_input_frames
        if self.input_start == 'contact':
            input_start = 0
        elif self.input_start == 'charpose':
            input_start = max(0, num_possible_shifts)
        elif self.input_start == 'random' and self.phase == 'train':
            input_start = np.random.choice(list(range(max(1, num_possible_shifts))), size=[1], replace=True)
        elif self.input_start == 'middle' or self.phase == 'val':
            input_start = max(0, num_possible_shifts // 2)
        elif self.input_start == 'onethird':
            input_start = max(0, num_possible_shifts // 3)
        elif self.input_start == 'twothirds':
            input_start = max(0, (num_possible_shifts // 3) * 2)
        else:
            raise ValueError

        # Preparing input and target sequences
        seq = pose_sequence - pose_sequence[:, [8]]
        input_frames = np.arange(contact_frame_idx + input_start * frames_per_pose_input, contact_frame_idx + (input_start + num_input_frames) * frames_per_pose_input, frames_per_pose_input)
        target_frames = [charpose_frame_idx]
        input_skeletons = np.stack([seq[frame] for frame in input_frames])
        target_skeleton = np.stack([seq[frame] for frame in target_frames])

        # Rotational augmentation around z axis
        if self.phase == 'train' and self.conf.augmentation_angle > 0:
            r = R.from_euler('z', np.random.random(size=[1]) * self.conf.augmentation_angle * 2 - self.conf.augmentation_angle, degrees=True)
            rotmat = r.as_matrix().squeeze(0) if r.as_matrix().ndim > 2 else r.as_matrix()
            input_skeletons = (rotmat[None, :] @ input_skeletons.transpose(0, 2, 1)).transpose(0, 2, 1)
            target_skeleton = (rotmat[None, :] @ target_skeleton.transpose(0, 2, 1)).transpose(0, 2, 1)

        output = [
            input_skeletons.astype(np.float32),
            target_skeleton.squeeze(0).astype(np.float32),
            sample_id
        ]

        return output
