from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pymp

from characteristic3dposes.ops.loss import mpjpe as calc_mpjpe
from characteristic3dposes.data import grab, h36m
from characteristic3dposes.data.grab.constants import Skeleton25
from characteristic3dposes.data.h36m.constants import Skeleton17
from characteristic3dposes.ops.render import render_skeleton


def render_skeletons(path: Path, sample_ids: list, all_samples=False, dataset_type='grab', refined=False):
    """
    Render input, predicted, and target skeletons using pyrender

    :param path: Path containing outputs from sample.py: results/ and, if refined, refined/
    :param sample_ids: List of samples IDs to be evaluated
    :param all_samples: Whether to render all samples per sample ID or just the top-1 one
    :param dataset_type: The dataset type
    :param refined: Whether to use the refined skeletons
    :return:
    """

    num_body_joints = 25 if dataset_type == 'grab' else 17

    results_path = path / 'results'
    renders_path = path / 'render'
    Path.mkdir(renders_path, exist_ok=True)

    with pymp.Parallel(num_threads=8, if_=False) as p:
        for idx, sample_id in p.iterate(enumerate(sample_ids)):
            p.print(f"[{idx + 1:03}/{len(sample_ids):03}] {sample_id}")

            # Load skeletons
            input_skeleton = np.load(str(results_path / f"{sample_id}-skeleton_input.npy")).reshape(num_body_joints, 3)
            predicted_skeleton = np.load(
                str(path / 'refined' / f"{sample_id}-skeleton-refinement.npy") if refined else
                str(results_path / f"{sample_id}-skeleton_prediction.npy")
            ).reshape(num_body_joints, 3)
            target_skeleton = np.load(str(results_path / f"{sample_id}-skeleton_target.npy")).reshape(num_body_joints, 3)

            # Load all samples
            if refined:
                if Path.is_file(path / 'refined' / f"{sample_id}-skeleton-samples_per_joint-refinement.npy"):
                    samples_per_joint = np.load(str(path / 'refined' / f"{sample_id}-skeleton-samples_per_joint-refinement.npy")).reshape(10, num_body_joints, 3)
            else:
                if Path.is_file(results_path / f"{sample_id}-samples_per_joint.npy"):
                    samples_per_joint = np.load(str(results_path / f"{sample_id}-samples_per_joint.npy"))

            # Render skeletons with pyrender
            render_skeleton(
                skeleton=input_skeleton + (np.array([0, 0, 1]).reshape(1, 3)),
                out_path=renders_path / f"{sample_id}-{'original' if not refined else 'refined'}-input.png",
                skeleton_cls=Skeleton25 if dataset_type == 'grab' else Skeleton17
            )
            render_skeleton(
                skeleton=predicted_skeleton + np.array([0, 0, 1]).reshape(1, 3),
                out_path=renders_path / f"{sample_id}-{'original' if not refined else 'refined'}-prediction.png",
                skeleton_cls=Skeleton25 if dataset_type == 'grab' else Skeleton17
            )
            render_skeleton(
                skeleton=target_skeleton + np.array([0, 0, 1]).reshape(1, 3),
                out_path=renders_path / f"{sample_id}-{'original' if not refined else 'refined'}-target.png",
                skeleton_cls=Skeleton25 if dataset_type == 'grab' else Skeleton17
            )

            # Also render all samples, if selected
            if all_samples:
                for sample_idx in range(10):
                    predicted_skeleton = samples_per_joint[sample_idx].reshape(num_body_joints, 3)

                    render_skeleton(
                        skeleton=predicted_skeleton + np.array([0, 0, 1]).reshape(1, 3),
                        out_path=renders_path / f"{sample_id}-{'original' if not refined else 'refined'}-prediction-sample-{sample_idx + 1}.png",
                        skeleton_cls=Skeleton25 if dataset_type == 'grab' else Skeleton17
                    )


def diversity(path: Path, sample_ids: list, k=10, dataset_type='grab', refined=False):
    """
    Calculate the diversity between all k sampled predictions

    :param path: Path containing outputs from sample.py: results/ and, if refined, refined/
    :param sample_ids: List of samples IDs to be evaluated
    :param k: Number of samples per sample id
    :param dataset_type: The dataset type
    :param refined: Whether to use the refined skeletons
    :return: The mean diversity over all samples
    """

    diversities = []
    for sample_id in sample_ids:
        samples = np.load(
            str(path / ('results' if not refined else 'refined') / (f"{sample_id}-samples_per_joint.npy"
                                                                    if not refined else
                                                                    f"{sample_id}-skeleton-samples_per_joint-refinement.npy")))

        mpjpes = []
        for i in range(k):
            for j in range(i+1, k):
                mpjpes.append(calc_mpjpe(samples[i], samples[j])[0])
        diversity = np.mean(np.stack(mpjpes), axis=0)
        diversities.append(diversity)

    diversity = np.mean(np.stack(diversities).mean(0) * (
        np.array([1, 1, 3, 4, 5, 3, 4, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        if dataset_type == 'grab' else
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2])))

    print(f"diversity_per_joint = {np.stack(diversities).mean(0)}")
    print(f"diversity = {diversity}")
    with open(path / 'diversity.txt', 'w') as f:
        f.write(f"diversity_per_joint = {np.stack(diversities).mean(0)}")
        f.write(f"diversity = {diversity}")

    return diversity


def mpjpe_eval(path: Path, sample_ids: list, dataset_type='grab', refined=False):
    """
    Calculate the top-1 mpjpe

    :param path: Path containing outputs from sample.py: results/ and, if refined, refined/
    :param sample_ids: List of samples IDs to be evaluated
    :param dataset_type: The dataset type
    :param refined: Whether to use the refined skeletons
    :return: The mean mpjpe over all samples
    """

    num_body_joints = 25 if dataset_type == 'grab' else 17
    mpjpes = np.zeros(shape=[len(sample_ids), num_body_joints])
    for sidx, sample_id in enumerate(sample_ids):
        predicted_skeleton = np.load(str(path / 'results' / f"{sample_id}-skeleton_prediction.npy")).reshape(num_body_joints, 3) if not refined else np.load(str(path / 'refined' / f"{sample_id}-skeleton-refinement.npy")).reshape(num_body_joints, 3)
        target_skeleton = np.load(str(path / 'results' / f"{sample_id}-skeleton_target.npy")).reshape(num_body_joints, 3)

        predicted_skeleton -= predicted_skeleton[8 if dataset_type == 'grab' else 0]
        target_skeleton -= target_skeleton[8 if dataset_type == 'grab' else 0]

        mpjpes[sidx] = calc_mpjpe(predicted_skeleton, target_skeleton)

    mpjpes_per_joint = mpjpes.mean(axis=0)
    mpjpe_overall = mpjpes.mean()

    print(f"mpjpes per_joint = {mpjpes_per_joint}")
    print(f"mpjpe overall = {mpjpe_overall}")
    with open(path / 'mpjpe.txt', 'w') as f:
        f.write(f"mpjpes per_joint = {mpjpes_per_joint}")
        f.write(f"mpjpe overall = {mpjpe_overall}")

    return mpjpe_overall


def main(experiment: str, input_start: str, dataset_type: str, refined=False):
    """
    Start evaluation: Qualitatively by rendering skeleton predictions; quantitatively by calculating mpjpe and diversity
    :param experiment: The experiment (grab or h36m)
    :param input_start: The input start (contact, onethird, middle, twothirds, charpose)
    :param dataset_type: The dataset type (grab or h36m)
    :param refined: Whether to use the refined skeletons
    :return: The overall mpjpe, diversity
    """
    assert input_start in ['contact', 'onethird', 'middle', 'twothirds', 'charpose']
    assert dataset_type in ['grab', 'h36m']
    assert experiment in ['grab', 'h36m']

    # Retrieve path to current step
    base = Path(__file__).parent.parent / 'experiments'
    path = next(Path.glob(base / experiment / 'evaluation', 'epoch=*-nsamples=10')) / input_start
    print(path)

    # Collect sample ids
    sample_ids = sorted(grab.sample_ids(containing='s10-') if dataset_type == 'grab' else h36m.sample_ids(containing='S5-'))

    # Quantitative Evaluation
    print("Diversity Evaluation")
    diversity_overall = diversity(path, sample_ids, dataset_type=dataset_type, refined=refined)
    print("MPJPE Evaluation")
    mpjpe_overall = mpjpe_eval(path, sample_ids, dataset_type=dataset_type, refined=refined)

    # Qualitative Evaluation
    print("Rendering")
    render_skeletons(path, sample_ids, all_samples=True, dataset_type=dataset_type, refined=refined)

    return mpjpe_overall, diversity_overall


if __name__ == '__main__':
    experiment = 'grab'  # grab, h36m

    parser = ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, default=experiment)
    parser.add_argument('-r', '--refined', action='store_true')
    args = parser.parse_args()

    mpjpes = []
    diversities = []

    # Loop over all five possible input sequence starts
    for input_start in ['contact', 'onethird', 'middle', 'twothirds', 'charpose']:
        print(f"Input start {input_start}")
        mpjpe_overall, diversity_overall = main(
            experiment=args.experiment,
            input_start=input_start,
            dataset_type=args.experiment,
            refined=args.refined
        )
        mpjpes.append(mpjpe_overall)
        diversities.append(diversity_overall)

    print(f"Results: MPJPE = [{mpjpes}], mean = {np.mean(mpjpes)}")
    print(f"Results: Diversity = [{diversities}], mean = {np.mean(diversities)}")
