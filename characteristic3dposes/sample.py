import re
from pathlib import Path
from types import SimpleNamespace
from typing import Union, Dict
from argparse import ArgumentParser

import numpy as np
import torch

from characteristic3dposes.model.main import Characteristic3DPosesModel
from characteristic3dposes.data.grab.dataset import GRABCharacteristicPoseDataset
from characteristic3dposes.data.h36m.dataset import H36MCharacteristicPoseDataset
from characteristic3dposes.ops.loss import MeanPerJointPositionError
from characteristic3dposes.ops.sampling import scaling_fun, JointSampler


@torch.no_grad()
def sample_steps(models: Dict[str, Union[GRABCharacteristicPoseDataset, H36MCharacteristicPoseDataset]],
                 data: torch.utils.data.DataLoader,
                 train_conf: SimpleNamespace, pose_criterion, sampler: JointSampler, output_path: Path):
    """
    Sample from predicted heatmaps, sample by sample

    :param models: A dict of models for right hand, left hand, and rest of the body; checkpoint weights already applied
    :param data: The data loader for the current dataset in test mode
    :param train_conf: The configuration used for training the model(s)
    :param pose_criterion: The pose criterion to determine the top-1 skeleton prediction
    :param sampler: The sampler, generating 3d xyz joint locations from heatmaps and offsets
    :param output_path: The path to write the outputs to
    :return:
    """

    # Constants: Joint indices for right and left hand, based on dataset type
    dataset_type = train_conf.data.type
    right_hand = 4 if dataset_type == 'grab' else 16
    left_hand = 7 if dataset_type == 'grab' else 13
    [model.eval() for model in models.values()]

    # Loop over all samples one-by-one
    for sample_idx, sample in enumerate(data):
        input_skeletons, target_skeleton, sample_ids = [elem.cuda() if type(elem) == torch.Tensor else elem for elem in sample]
        print(f"[{sample_idx+1:02}/{len(data)}] Sampling {sample_ids[0]}")

        predicted_skeleton = torch.zeros(size=[17 if dataset_type == 'h36m' else 25, 10, 3], device=target_skeleton.device)
        predicted_heatmaps = torch.zeros(size=[17 if dataset_type == 'h36m' else 25, 10, 16, 16, 16], device=target_skeleton.device)
        scaled_heatmaps = torch.zeros(size=[17 if dataset_type == 'h36m' else 25, 16, 16, 16], device=target_skeleton.device)

        # Right Hand
        rh_model_key, rh_model = list(models.items())[0]
        assert 'rh' in rh_model_key
        rh_prediction = rh_model(input_skeletons, [right_hand], previous_joints=[])
        rh_predicted_heatmap = rh_prediction['heatmap'] if train_conf.training.heatmap.criterion != 'ce' else torch.argmax(rh_prediction['heatmap'], dim=2).float().unsqueeze(2) / float(train_conf.training.heatmap.num_classes)

        scaled_heatmap = scaling_fun(rh_predicted_heatmap, strategy='simple').view(1, 16, 16, 16)
        scaled_heatmap = (scaled_heatmap / scaled_heatmap.sum()) if scaled_heatmap.sum() > 0 else scaled_heatmap
        predicted_heatmaps[right_hand] = rh_prediction['heatmap']
        scaled_heatmaps[right_hand] = scaled_heatmap

        rh_joint_samples = sampler(scaled_heatmap.squeeze(), rh_prediction['offsets'].squeeze(), k=5, nms=True)
        predicted_skeleton[right_hand] = rh_joint_samples.repeat_interleave(2, 0)

        # Left Hand
        lh_model_key, lh_model = list(models.items())[1]
        assert 'lh' in lh_model_key
        lh_joint_samples = []
        for joint_sample_idx, joint_sample in enumerate(rh_joint_samples):
            lh_prediction = lh_model(input_skeletons, [left_hand], previous_joints=[joint_sample.unsqueeze(0).unsqueeze(0)])

            lh_predicted_heatmap = lh_prediction['heatmap'] if train_conf.training.heatmap.criterion != 'ce' else torch.argmax(lh_prediction['heatmap'], dim=2).float().unsqueeze(2) / float(train_conf.training.heatmap.num_classes)
            scaled_heatmap = scaling_fun(lh_predicted_heatmap, strategy='simple').view(1, 16, 16, 16)
            scaled_heatmap = (scaled_heatmap / scaled_heatmap.sum()) if scaled_heatmap.sum() > 0 else scaled_heatmap
            predicted_heatmaps[left_hand] = lh_prediction['heatmap']
            scaled_heatmaps[left_hand] = scaled_heatmap

            lh_joint_sample = sampler(scaled_heatmap.squeeze(), lh_prediction['offsets'].squeeze(), k=2, nms=True)
            lh_joint_samples.append(lh_joint_sample.squeeze(0))
            predicted_skeleton[left_hand, joint_sample_idx*2:(joint_sample_idx+1)*2] = lh_joint_sample

        # Rest of the body
        main_model_key, main_model = list(models.items())[2]
        assert 'rest' in main_model_key
        for rh_lh_joint_sample_idx, rh_lh_joint_sample in enumerate(zip(predicted_skeleton[right_hand], predicted_skeleton[left_hand])):
            prediction = main_model(input_skeletons, train_conf.training.joint.predict, previous_joints=[elem.unsqueeze(0).unsqueeze(0) for elem in rh_lh_joint_sample])
            predicted_heatmap = prediction['heatmap'] if train_conf.training.heatmap.criterion != 'ce' else torch.argmax(prediction['heatmap'], dim=2).float().unsqueeze(2) / float(train_conf.training.heatmap.num_classes)

            for joint_idx, joint in enumerate(train_conf.training.joint.predict):
                scaled_heatmap = scaling_fun(predicted_heatmap[:, joint_idx], strategy='simple', temperature=0.01).view(1, 16, 16, 16)
                scaled_heatmap = (scaled_heatmap / scaled_heatmap.sum()) if scaled_heatmap.sum() > 0 else scaled_heatmap
                predicted_heatmaps[joint] = prediction['heatmap'][:, joint_idx]
                scaled_heatmaps[joint] = scaled_heatmap

                samples = sampler(scaled_heatmap.squeeze(), prediction['offsets'][:, [joint_idx]].squeeze(), k=1)
                predicted_skeleton[joint, rh_lh_joint_sample_idx:rh_lh_joint_sample_idx+1] = samples

        predicted_skeleton_best = predicted_skeleton[:, torch.argmin(torch.stack([pose_criterion(predicted_skeleton[:, idx], target_skeleton).mean(1) for idx in range(10)])).item()]

        np.save(str(output_path / f"{sample_ids[0]}-skeleton_input"), input_skeletons[:, -1].cpu().numpy())
        np.save(str(output_path / f"{sample_ids[0]}-skeletons_input"), input_skeletons.cpu().numpy())
        np.save(str(output_path / f"{sample_ids[0]}-skeleton_prediction"), predicted_skeleton_best.cpu().numpy())
        np.save(str(output_path / f"{sample_ids[0]}-samples_per_joint"), predicted_skeleton.transpose(1, 0).cpu().numpy())
        np.save(str(output_path / f"{sample_ids[0]}-skeleton_target"), target_skeleton.cpu().numpy())
        np.save(str(output_path / f"{sample_ids[0]}-volumes_predicted_scaled"), scaled_heatmaps.cpu().numpy())
        np.save(str(output_path / f"{sample_ids[0]}-volumes_predicted"), predicted_heatmaps.cpu().numpy())


def main(experiment: str, base_path: Path):
    """
    Start sampling given experiment (either grab or h36m) with the given base_path

    :param experiment: The experiment to sample: grab or h36m, has to be trained in base_path / experiment
    :param base_path: The base path, usually the parent of the parent of this file
    :return:
    """

    assert experiment in ['grab', 'h36m']

    print(f"Sampling in base path {base_path}")

    # Initialize models with best checkpoints
    models = {}
    for model_key_idx, model_key in enumerate(['rh', 'lh', 'rest']):
        # Get list of all model checkpoint files
        base_model_path = base_path / experiment / f'{model_key_idx + 1}' / 'models'
        model_files = list(base_model_path.iterdir())
        # Load train configuration
        train_conf = torch.load(model_files[0], map_location=torch.device('cuda'))['train_conf']
        # Load best model checkpoint file
        r = re.compile('epoch=(.+)-loss=(.*)')
        best_model_path = sorted([(float(r.match(file.stem).groups()[1]), file) for file in model_files], key=lambda x: x[0])[0][1]
        models[model_key] = Characteristic3DPosesModel(train_conf).cuda()
        models[model_key].load_state_dict(torch.load(best_model_path, map_location=torch.device('cuda'))['model'])

    # Misc
    prefix = f'{best_model_path.stem}' + f'-nsamples={10}'
    sampler = JointSampler(heatmap_resolution=16, heatmap_sigma=int(train_conf.training.heatmap.sigma)).cuda()
    pose_criterion = MeanPerJointPositionError(keep_joints=True).cuda()

    for input_start in ['contact', 'onethird', 'middle', 'twothirds', 'charpose']:
        print(f"Sampling for input start {input_start}")

        # Output path for this input start
        output_path = base_path / experiment / 'evaluation' / prefix / input_start / 'results'
        Path.mkdir(output_path, exist_ok=True, parents=True)

        # Initialize dataloader in test mode
        dataset_cls = GRABCharacteristicPoseDataset if train_conf.data.type == 'grab' else H36MCharacteristicPoseDataset
        test_data = torch.utils.data.DataLoader(
            dataset_cls(phase='test',
                        conf=SimpleNamespace(file=train_conf.data.file, type=train_conf.data.type,
                                             overfit=False, multiplicator=1, augmentation_angle=0),
                        input_start=input_start),
                        shuffle=False, batch_size=1, num_workers=8, pin_memory=True, drop_last=False)

        # Sample each sample individually
        sample_steps(models, test_data, train_conf, pose_criterion, sampler, output_path)


if __name__ == '__main__':
    # The experiment to evaluate. Choose between grab and h36m here.
    # Your model should be trained in the corresponding directory in experiments/
    experiment = 'grab'  # grab, h36m

    parser = ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, default=experiment)
    args = parser.parse_args()

    main(experiment=args.experiment, base_path=Path(__file__).parent.parent / 'experiments')
