from pathlib import Path

import numpy as np
import pandas as pd

import cdflib


SUBJECTS = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']


def main(input_path: Path, output_path: Path):
    """
    Generate the characteristic 3d pose data from the Human3.6M dataset
    Instruction on how to download and extract Human3.6M data are in README.md

    :param input_path: The path containing all 3d pose files downloaded from the Human3.6M website. Expected to contain diretories S1/, S5/, S6/, S7/, S8/, S9/, S11
    :param output_path: Where the final numpy npz file will be stored
    :return:
    """

    output_path.mkdir(exist_ok=True)

    pose_sequences = {}
    charposes = {}
    charpose_indices = {}

    # Load annotations
    annotations = pd.read_csv(Path(__file__).parent / 'charposes.csv', delimiter=';')
    annotations = {elem[0].replace('TakingPhoto', 'Photo').replace('WalkingDog', 'WalkDog'): (elem[1], elem[2]) for elem in zip(list(annotations['sample-id']), list(annotations['start']), list(annotations['action']))}

    # Load each sequence and extract 3d poses at the annotated time stamps
    print("Generating ...")
    for subject in SUBJECTS:
        pose_sequences[subject] = {}
        charposes[subject] = {}
        charpose_indices[subject] = {}

        files = list((input_path / subject / 'MyPoseFeatures' / 'D3_Positions').glob('*.cdf'))
        assert len(files) == 30, "Expected 30 files for subject " + subject + ", got " + str(len(files))

        for file in files:
            action = file.stem.replace(' ', '_')
            action = action.replace('TakingPhoto', 'Photo').replace('WalkingDog', 'WalkDog')

            sample_id = f'{subject}-{action}'
            print(f"|-> {sample_id}")

            cf = cdflib.CDF(str(file))
            pose_sequence = cf['Pose'].reshape(-1, 32, 3) / 1000.
            cf.close()

            pose_index_dict = {
                'start': annotations[sample_id][0],
                'charpose': annotations[sample_id][1],
                'end': len(pose_sequence) - 1
            }

            poses = np.stack([pose_sequence[idx] for idx in pose_index_dict.values()])
            pose_indices = np.stack(list(pose_index_dict.values()))

            pose_sequences[subject][action] = pose_sequence.astype('float32')
            charposes[subject][action] = poses.astype('float32')
            charpose_indices[subject][action] = pose_indices

    # Write npz
    print("Saving ...")
    np.savez(str(output_path / 'data_3d_h36m'), pose_sequences=pose_sequences, charposes=charposes, charpose_indices=charpose_indices)


if __name__ == '__main__':
    # You can change your input and output paths but this configuration works without further modifications
    main(
        input_path=Path(__file__).parent / 'input',
        output_path=Path(__file__).parent / 'output'
    )
