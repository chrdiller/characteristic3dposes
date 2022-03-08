from argparse import ArgumentParser
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pymp
from jax import grad, hessian
from scipy.optimize import minimize

from characteristic3dposes.data.grab.constants import Skeleton25
from characteristic3dposes.data.h36m.constants import Skeleton17
from characteristic3dposes.ops.loss import mpjpe
from characteristic3dposes.data import grab, h36m

jax.config.update('jax_enable_x64', True)


def joints_to_heatmap_indices(joints):
    return jax.lax.clamp(0, jnp.floor(((joints + 1.) * 16 / 2.)).astype(jnp.int64), 16 - 1)


def cos_angle_between(v1, v2):
    return jnp.dot(v1, v2) / (jnp.linalg.norm(v1 + 1.) * jnp.linalg.norm(v2 + 1.))


def calculate_bone_lengths(joints, skeleton_cls):
    return jnp.array([jnp.linalg.norm(joints[bone[0]] - joints[bone[1]] + 0.001) for bone in skeleton_cls.bones])


def calculate_cos_angles(joints, skeleton_cls):
    return jnp.array([cos_angle_between(joints[angle[0]] - joints[angle[1]], joints[angle[2]] - joints[angle[1]]) for angle in skeleton_cls.angle_joints])


def heatmap_error(joints, heatmap, skeleton_cls):
    heatmap_indices = joints_to_heatmap_indices(joints)
    return jnp.sum(jnp.array(skeleton_cls.joints_weights) * (1. - jnp.array([heatmap[joint_idx][joint[0], joint[1], joint[2]] for joint_idx, joint in enumerate(heatmap_indices)])))


def bone_length_error(joints, input_bone_lengths, skeleton_cls):
    bone_lengths = calculate_bone_lengths(joints, skeleton_cls)
    return jnp.sum(jnp.abs(jnp.array(input_bone_lengths) - bone_lengths))


def angle_error(joints, input_cos_angles, skeleton_cls):
    cos_angles = calculate_cos_angles(joints, skeleton_cls)
    return jnp.sum(jnp.array(skeleton_cls.angle_weights) * jnp.abs(input_cos_angles - cos_angles))


def end_effector_error(joints, end_effectors, skeleton_cls):
    return jnp.linalg.norm(joints[(skeleton_cls.end_effectors[0], skeleton_cls.end_effectors[1]), :] - end_effectors + 0.01)


def closeness_term(joints, input_joints):
    distances = jnp.abs(jnp.array(joints) - jnp.array(input_joints))
    return jnp.sum(jnp.array(distances))


def objective_func(x, end_effectors, input_skeleton, heatmap, input_bone_lengths, input_cos_angles, weights, skeleton_cls):
    joints = x.reshape(skeleton_cls.num_joints, 3)

    error = \
        weights[0] * end_effector_error(joints, end_effectors, skeleton_cls) + \
        weights[1] * bone_length_error(joints, input_bone_lengths, skeleton_cls) + \
        weights[2] * angle_error(joints, input_cos_angles, skeleton_cls) + \
        (weights[3] * heatmap_error(joints, heatmap, skeleton_cls) if weights[3] > 0 else 0) + \
        weights[4] * closeness_term(joints, input_skeleton)

    return error


def refine_skeleton(predicted_skeleton: np.array, input_skeleton: np.array, heatmap: np.array, weights: list, skeleton_cls):
    """
    Refine a predicted skeleton, based on its input skeleton, the heatmap, and the given weights
    :param predicted_skeleton: The predicted skeleton, to be refined
    :param input_skeleton: The last input skeleton
    :param heatmap: The heatmap the predicted skeleton was sampled from
    :param weights: The weights for the optimization
    :param skeleton_cls: The skeleton class, used for bones, weights, and joint angle definitions
    :return: The refined skeleton
    """

    x0 = predicted_skeleton

    # Get input bone lengths and end effector locations
    input_bone_lengths = calculate_bone_lengths(input_skeleton, skeleton_cls)
    input_cos_angles = calculate_cos_angles(input_skeleton, skeleton_cls)
    end_effectors = np.stack([x0[skeleton_cls.end_effectors[0]], x0[skeleton_cls.end_effectors[1]]])

    # Calculate grad and hessian of objective function
    objective_func_grad = grad(objective_func)
    objective_func_hess = hessian(objective_func)

    # Optimize the predicted skeleton
    optimized_skeleton = minimize(objective_func, x0=x0.flatten(),
                                  args=(end_effectors, input_skeleton, heatmap, input_bone_lengths, input_cos_angles, weights, skeleton_cls),
                                  jac=lambda x, a1, a2, a3, a4, a5, a6, a7: np.array(objective_func_grad(x, a1, a2, a3, a4, a5, a6, a7)),
                                  hess=lambda x, a1, a2, a3, a4, a5, a6, a7: np.array(objective_func_hess(x, a1, a2, a3, a4, a5, a6, a7)),
                                  options={'maxiter': 1000, 'disp': True}, method='Newton-CG')

    return optimized_skeleton.x.reshape(skeleton_cls.num_joints, 3), optimized_skeleton.success, optimized_skeleton.message


def main(path: Path, input_start: str, parallel=8, weights=None, all_samples=True, dataset_type='grab'):
    """
    Run refinement of which will improve the quality of predicted skeletons by enforcing constraints on bone lengths,
    joint angles, end effector placements, and closeness to the input

    :param path: Path containing outputs from sample.py: results/. Will place outputs into directory refined/ here
    :param input_start: The input start to refine
    :param parallel: How many processes to start in parallel. Each process optimizes one sample. Uses pymp for load distribution. Set to 0 to disable parallel processing (e.g. for debugging)
    :param weights: The weights for [end_effector_error, bone_length_error, angle_error, heatmap_error, input_closeness_error]
    :param all_samples: Whether to refine all k samples per sample id or just the top-1 skeleton
    :param dataset_type: Dataset type, either grab or h36m
    :return:
    """

    assert dataset_type in ['grab', 'h36m']

    path = next(Path.glob(path, 'epoch=*-nsamples=10')) / input_start
    print(path)
    results_path = path / 'results'
    output_path = path / 'refined'
    output_path.mkdir(exist_ok=True)

    skeleton_cls = Skeleton25 if dataset_type == 'grab' else Skeleton17
    sample_ids = (grab.sample_ids(containing='s10-') if dataset_type == 'grab' else h36m.sample_ids(containing='S5-'))

    results = pymp.shared.list()
    with pymp.Parallel(parallel, if_=(parallel > 0)) as p:
        for sidx, sample_id in p.iterate(enumerate(sample_ids)):
            input_skeleton = np.load(str(results_path / f"{sample_id}-skeleton_input.npy"))[-1].reshape(skeleton_cls.num_joints, 3)
            predicted_skeleton = np.load(str(results_path / f"{sample_id}-skeleton_prediction.npy")).reshape(skeleton_cls.num_joints, 3)
            target_skeleton = np.load(str(results_path / f"{sample_id}-skeleton_target.npy")).reshape(skeleton_cls.num_joints, 3)
            scaled_heatmaps = np.load(str(results_path / f"{sample_id}-volumes_predicted_scaled.npy")).reshape(skeleton_cls.num_joints, 16, 16, 16) if Path.is_file(results_path / f"{sample_id}-volumes_predicted_scaled.npy") else None

            weights[-1] = 0. if mpjpe(input_skeleton, predicted_skeleton)[0, skeleton_cls.left_leg+skeleton_cls.right_leg].mean() > 0.04 else weights[-1]
            refined_skeleton, success, message = refine_skeleton(predicted_skeleton, input_skeleton, scaled_heatmaps, weights, skeleton_cls)

            mpjpe_before = mpjpe(predicted_skeleton, target_skeleton)
            mpjpe_after = mpjpe(refined_skeleton, target_skeleton)

            with p.lock:
                results.append({'sample_id': sample_id, 'success': success, 'message': message, 'mpjpe_before': mpjpe_before, 'mpjpe_after': mpjpe_after if mpjpe_after.mean() < mpjpe_before.mean() else mpjpe_before})
                p.print(f"[{sidx+1}/{len(sample_ids)}] {sample_id}: {mpjpe_before.mean():.5f} -> {mpjpe_after.mean():.5f} ({(1. - mpjpe_after.mean()/mpjpe_before.mean()) * 100.:.2f}% reduction)")

            np.save(str(output_path / f"{sample_id}-skeleton-refinement"), refined_skeleton)

            if all_samples:
                p.print(f"[{sidx+1}/{len(sample_ids)}] Refining all samples ...")
                try:
                    predicted_skeleton_samples = np.load(str(results_path / f"{sample_id}-samples_per_joint.npy")).reshape(10, skeleton_cls.num_joints, 3)
                    refined_skeleton_samples = []
                    for sample_idx, predicted_skeleton in enumerate(predicted_skeleton_samples):
                        p.print(f"[{sidx + 1}/{len(sample_ids)}] Sample {sample_idx+1} ...")
                        refined_skeleton, success, message = refine_skeleton(predicted_skeleton, input_skeleton, scaled_heatmaps, weights, skeleton_cls)
                        refined_skeleton_samples.append(refined_skeleton)
                    np.save(str(output_path / f"{sample_id}-skeleton-samples_per_joint-refinement.npy"), np.stack(refined_skeleton_samples))
                except ValueError:
                    print('Skipping')

    pd.DataFrame({
        'sample_id': [elem['sample_id'] for elem in results],
        'success': [elem['success'] for elem in results],
        'message': [elem['message'] for elem in results]}
    ).to_csv(output_path / f'refinement-result.csv')

    per_sample_mpjpe_before = np.concatenate([elem['mpjpe_before'] for elem in results]).mean(axis=1)
    per_sample_mpjpe_after = np.concatenate([elem['mpjpe_after'] for elem in results]).mean(axis=1)

    print(f"Before = {per_sample_mpjpe_before.mean()}, After = {per_sample_mpjpe_after.mean()}")
    print(f"Overall refinement improvement = {per_sample_mpjpe_before.mean() - per_sample_mpjpe_after.mean()}")


if __name__ == '__main__':
    base_path = Path(__file__).parent.parent / 'experiments'
    experiment = 'grab'  # grab, h36m

    parser = ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, default=experiment)
    parser.add_argument('-p', '--parallel', type=int, default=8)
    parser.add_argument('-a', '--all_samples', action='store_true')
    args = parser.parse_args()

    weights = [0.2, 1., 0.4, 0.1, 1.]

    for input_start in ['contact', 'onethird', 'middle', 'twothirds', 'charpose']:
        main(
            path=base_path / args.experiment / 'evaluation',
            input_start=input_start,
            parallel=args.parallel,
            weights=weights,
            all_samples=args.all_samples,
            dataset_type=args.experiment,
        )
