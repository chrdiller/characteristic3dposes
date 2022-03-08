import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import hydra
import omegaconf
import numpy as np
import torch

from characteristic3dposes.model.main import Characteristic3DPosesModel
from characteristic3dposes.data.grab.dataset import GRABCharacteristicPoseDataset
from characteristic3dposes.data.h36m.dataset import H36MCharacteristicPoseDataset
from characteristic3dposes.ops.loss import PoseHeatMapCriterion, HeatmapOffsetCriterion
from characteristic3dposes.ops.misc import FixedWithWarmupOptimizer, summarize_model


def train_epoch(model: Characteristic3DPosesModel, data: torch.utils.data.DataLoader,
                heatmap_criterion: PoseHeatMapCriterion, heatmap_criterion_small: PoseHeatMapCriterion,
                offset_criterion: HeatmapOffsetCriterion, optimizers: dict,
                conf: SimpleNamespace, epoch: int, last_val_loss: float):
    """
    Training, once every epoch

    :param model: The model being trained
    :param data: The dataloader, in train mode
    :param heatmap_criterion: The heatmap criterion
    :param heatmap_criterion_small: The heatmap criterion, for the intermediate, small heatmap
    :param offset_criterion: The offset criterion
    :param optimizers: The dict of optimizers, containing keys offsets, heatmap, encoder
    :param conf: The train configuration, as parsed by hydra
    :param epoch: The current epoch
    :param last_val_loss: The val loss of last epoch's validation run, as additional output info
    :return:
    """

    model.train()

    # Only train offsets prediction after a set number of epochs
    if epoch > conf.training.offsets.after_epoch:
        for param in model.parameters(recurse=True):
            param.requires_grad = False
        for param in model.offsets_head.parameters(recurse=True):
            param.requires_grad = True
        for param in model.offsets_tail.parameters(recurse=True):
            param.requires_grad = True

    for batch_idx, batch in enumerate(data):
        input_skeletons, target_skeleton, sample_ids = [elem.cuda() if type(elem) == torch.Tensor else elem for elem in batch]

        # Forward
        [optimizer.zero_grad() for optimizer in optimizers.values()]
        prediction = model(
            input_skeletons,
            conf.training.joint.predict,
            previous_joints=[target_skeleton[:, [given_joint], :] for given_joint in conf.training.joint.given]
        )

        # Loss
        per_joint_heatmap_losses = []
        per_joint_offset_losses = []
        for joint_idx, joint in enumerate(conf.training.joint.predict):
            joint_heatmap_loss_full, target_heatmap = heatmap_criterion(prediction['heatmap'][:, joint_idx], target_skeleton[:, [joint], :], return_target_volume=True)
            joint_heatmap_loss_small = heatmap_criterion_small(prediction['heatmap_small'][:, joint_idx], target_skeleton[:, [joint], :], return_target_volume=False)

            joint_heatmap_loss = joint_heatmap_loss_small + joint_heatmap_loss_full
            per_joint_heatmap_losses.append(joint_heatmap_loss)
            joint_offset_loss = offset_criterion(prediction['offsets'][:, joint_idx], target_skeleton[:, [joint], :], prediction['heatmap'][:, joint_idx], target_heatmap)
            per_joint_offset_losses.append(joint_offset_loss)

        # Backward
        loss_heatmap = torch.stack(per_joint_heatmap_losses).mean()
        loss_offsets = torch.stack(per_joint_offset_losses).mean()
        loss = loss_heatmap if epoch <= conf.training.offsets.after_epoch else loss_offsets
        loss.backward()

        # Optimize
        if epoch > conf.training.offsets.after_epoch:
            optimizers['offsets'].step()
        else:
            optimizers['heatmap'].step()
            optimizers['encoder'].step_and_update_lr()

        print(f'[{epoch+1}/{conf.training.epochs} train | {batch_idx+1}/{len(data)}] total train = {loss.item():.5f} (heatmap = {loss_heatmap.item():.5f}, offsets = {loss_offsets.item():.5f}) | latest total val = {last_val_loss:.5f}')


def val_epoch(model: Characteristic3DPosesModel, data: torch.utils.data.DataLoader,
              heatmap_criterion: PoseHeatMapCriterion, offset_criterion: HeatmapOffsetCriterion,
              conf: SimpleNamespace, epoch: int):
    """
    Validation, once every epoch

    :param model: The model being trained
    :param data: The dataloader, in val mode
    :param heatmap_criterion: The heatmap criterion
    :param offset_criterion: The offset criterion
    :param conf: The train configuration, as parsed by hydra
    :param epoch: The current epoch
    :return: The mean val loss for this validation run
    """

    model.eval()

    val_results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(data):
            input_skeletons, target_skeleton, sample_ids = [elem.cuda() if type(elem) == torch.Tensor else elem for elem in batch]

            # Forward
            prediction = model(
                input_skeletons,
                conf.training.joint.predict,
                previous_joints=[target_skeleton[:, given_joint, :].unsqueeze(1) for given_joint in conf.training.joint.given]
            )

            # Loss
            logger_result = {}
            per_joint_heatmap_losses = []
            per_joint_offset_losses = []
            full_target_heatmap = []
            for joint_idx, joint in enumerate(conf.training.joint.predict):
                joint_heatmap_loss_full, target_heatmap = heatmap_criterion(prediction['heatmap'][:, joint_idx], target_skeleton[:, [joint], :], return_target_volume=True)
                joint_heatmap_loss = joint_heatmap_loss_full

                full_target_heatmap.append(target_heatmap)
                per_joint_heatmap_losses.append(joint_heatmap_loss)
                joint_offset_loss = offset_criterion(prediction['offsets'][:, joint_idx], target_skeleton[:, [joint], :], prediction['heatmap'][:, joint_idx], target_heatmap)
                per_joint_offset_losses.append(joint_offset_loss)

            loss_heatmap = torch.stack(per_joint_heatmap_losses).mean()
            loss_offsets = torch.stack(per_joint_offset_losses).mean() if epoch > conf.training.offsets.after_epoch else torch.tensor(0.)
            loss = loss_offsets if epoch > conf.training.offsets.after_epoch else (loss_heatmap + loss_offsets)

            logger_result.update({
                'heatmap': loss_heatmap.item(),
                'offsets': loss_offsets.item(),
                'loss': loss.item(),
            })

            val_results.append(logger_result)

            print(f'[{epoch+1}/{conf.training.epochs} val | {batch_idx + 1}/{len(data)}] val total = {loss.item():.5f} (heatmap = {loss_heatmap.item():.5f}, offsets = {loss_offsets.item():.5f})')

    mean_loss = np.stack([x['loss'] for x in val_results]).mean()
    return mean_loss


@hydra.main(config_path='.', config_name='config')
def main(conf):
    """
    Start training. The configuration can be found in config.yaml and will be parsed by hydra

    :param conf: The configuration, populated by hydra
    :return:
    """

    # Configuration
    print(f"Working directory: {Path.cwd()}")
    print("Configuration:")
    print(omegaconf.OmegaConf.to_yaml(conf))
    conf = json.loads(json.dumps(omegaconf.OmegaConf.to_container(conf)), object_hook=lambda d: SimpleNamespace(**d))

    # Paths
    model_path = Path.cwd() / 'models'
    if model_path.is_dir():
        shutil.rmtree(model_path)
    model_path.mkdir(parents=False, exist_ok=False)

    # Torch init
    torch.backends.cudnn.benchmark = True

    # Model
    model = Characteristic3DPosesModel(conf).cuda()
    print("Model:")
    print(summarize_model(model))

    # Criteria
    heatmap_criterion = PoseHeatMapCriterion(conf).cuda()
    heatmap_criterion_small = PoseHeatMapCriterion(conf, small=True).cuda()
    offset_criterion = HeatmapOffsetCriterion(conf).cuda()

    # Optimizers
    optimizers = {
        'offsets': torch.optim.AdamW(
            list(model.offsets_head.parameters(recurse=True)) + list(model.offsets_tail.parameters(recurse=True)),
            lr=conf.training.lr, weight_decay=conf.training.weight_decay),
        'heatmap': torch.optim.AdamW(
            list(model.heatmap_head.parameters(recurse=True)) + list(model.heatmap_tail.parameters(recurse=True)),
            lr=conf.training.lr, weight_decay=conf.training.weight_decay),
        'encoder': FixedWithWarmupOptimizer(torch.optim.AdamW(
            (list(model.previous_joint_encoder.parameters(recurse=True)) if 'previous_joint_encoder' in model.__dict__ else []) +
            list(model.pose_sequence_encoder.parameters(recurse=True)) +
            list(model.attention_model.parameters(recurse=True)),
            weight_decay=conf.training.weight_decay), final_lr=conf.training.lr, n_warmup_steps=conf.training.num_warmup_steps)
    }

    # Data
    train_data = torch.utils.data.DataLoader(
        GRABCharacteristicPoseDataset(phase='train', conf=conf.data) if conf.data.type == 'grab' else H36MCharacteristicPoseDataset(phase='train', conf=conf.data),
        shuffle=True,
        batch_size=conf.training.batch_size,
        num_workers=conf.training.workers,
        pin_memory=True, worker_init_fn=lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id))

    val_data = torch.utils.data.DataLoader(
        GRABCharacteristicPoseDataset(phase='val', conf=conf.data) if conf.data.type == 'grab' else H36MCharacteristicPoseDataset(phase='val', conf=conf.data),
        shuffle=False,
        batch_size=conf.training.batch_size,
        num_workers=conf.training.workers,
        pin_memory=True,
        drop_last=False, worker_init_fn=lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id))

    # Initial val check
    val_loss = val_epoch(model, val_data, heatmap_criterion, offset_criterion, conf, epoch=-1)

    # Training loop
    for epoch_idx in range(conf.training.epochs):
        if epoch_idx == conf.training.offsets.after_epoch + 1:
            print(f"[===>] Starting offset training now")
            checkpoints = [torch.load(str(path)) for path in model_path.iterdir()]
            best_checkpoint = checkpoints[np.argmin([checkpoint['val_loss'] for checkpoint in checkpoints])]

            if best_checkpoint['val_loss'] < val_loss:
                print(f"[===>] Loading best epoch {best_checkpoint['epoch']}: {best_checkpoint['val_loss']} (ckpt) < {val_loss} (ours)")
                model.load_state_dict(best_checkpoint['model'], strict=True)
                optimizers['offsets'].load_state_dict(best_checkpoint['optimizer_offsets'])
                optimizers['heatmap'].load_state_dict(best_checkpoint['optimizer_heatmap'])
                optimizers['encoder'].load_state_dict(best_checkpoint['optimizer_encoder'])
                val_loss = best_checkpoint['val_loss']
            else:
                print(f"[===>] Already best epoch: {best_checkpoint['val_loss']} (ckpt) >= {val_loss} (ours)")

        train_epoch(model, train_data, heatmap_criterion, heatmap_criterion_small, offset_criterion, optimizers, conf, epoch=epoch_idx, last_val_loss=val_loss)
        val_loss = val_epoch(model, val_data, heatmap_criterion, offset_criterion, conf, epoch=epoch_idx)

        # Checkpointing
        state_dict = {
            'epoch': epoch_idx,
            'optimizer_offsets': optimizers['offsets'].state_dict(),
            'optimizer_heatmap': optimizers['heatmap'].state_dict(),
            'optimizer_encoder': optimizers['encoder'].state_dict(),
            'train_conf': conf,
            'val_loss': val_loss,
            'model': model.state_dict()
        }
        torch.save(state_dict, str(model_path / f'epoch={epoch_idx}-loss={val_loss:.10f}.pth'))


if __name__ == '__main__':
    main()
