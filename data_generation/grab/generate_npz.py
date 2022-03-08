from pathlib import Path
import glob

import numpy as np
import smplx
import pandas as pd
import pymp

from data_generation.grab.grab.tools.meshviewer import Mesh
from data_generation.grab.grab.tools import utils as grab_utils


# Convert some non-standard sample ids into the default form and remove the outlier sample s1-doorknob-use-fun-1
CONVERT_SAMPLE_ID = {
    's6-bowl-drink-1-Retake': 's6-bowl-drink-2',
    's6-stapler-staple-1-Retake': 's6-stapler-staple-3',
    's6-piggybank-lift-Retake': 's6-piggybank-lift-1',
    's6-apple-offhand-1-Retake': 's6-apple-offhand-1',
    's6-toothpaste-squeeze-1-Retake': 's6-toothpaste-squeeze-1',
    's6-stapler-offhand-1-Retake': 's6-stapler-offhand-1',
    's3-cylinderlarge-lift-Retake': 's3-cylinderlarge-lift-1',
    's9-piggybank-lift-Retake': 's9-piggybank-lift-1',
    's9-wineglass-toast-1-Retake': 's9-wineglass-toast-1',
    's9-pyramidmedium-lift-Retake': 's9-pyramidmedium-lift-1',
    's1-camera-takepicture-3-Retake': 's1-camera-takepicture-3',
    's1-doorknob-use-fun-1': None,
    's2-piggybank-pass-1-Retake': 's2-piggybank-pass-1',
    's2-pyramidlarge-pass-1-Retake': 's2-pyramidlarge-pass-1',
    's7-waterbottle-open-1-Retake': 's7-waterbottle-open-1',
    's7-toruslarge-inspect-1-Retake': 's7-toruslarge-inspect-1',
    's7-airplane-lift-Retake': 's7-airplane-lift-1',
    's10-bowl-drink-1-Retake': 's10-bowl-drink-1',
    's10-alarmclock-lift-Retake': 's10-alarmclock-lift-1'
}


# Convert the SMPLX joint index (value) to the corresponding joint index in OpenPose layout (list index)
SKELETON_SMPLX2OPENPOSE_FINGERHAND = [55, 12, 17, 19, 42, 16, 18, 27, 0, 2, 5, 8, 1, 4, 7, 24, 23, 58, 59, 60, 61, 62, 63, 64, 65]


def main(input_path: Path, output_path: Path):
    """
    Generate a numpy npz file containing the xyz joint locations of all characteristic 3d poses in our dataset

    :param input_path: The path containing all files downloaded from the GRAB and SMPL-X websites; should be 13 .zip files in total
    :param output_path: Where the final numpy npz file will be stored
    :return:
    """

    output_path.mkdir(exist_ok=True)

    pose_sequences = pymp.shared.dict()
    charpose_indices = pymp.shared.dict()
    charposes = pymp.shared.dict()

    annotations = pd.read_csv(Path(__file__).parent / 'charposes.csv', delimiter=';')
    annotations = {elem[0]: elem[1] for elem in zip(list(annotations['sampleid']), list(annotations['frameidx']))}

    sequences = [seq for seq in glob.glob(str(input_path) + '/*/*.npz') if not 'smplx' in seq]

    # Extract skeletons with 8 processes simultaneously
    # You can change the number of processes here or set if_=False to disable parallelism
    # Expected runtime: About 20-30 minutes with 8 processes
    with pymp.Parallel(8, if_=False) as p:
        for sequence_idx, sequence in p.iterate(enumerate(sequences)):
            p.print(f"[{sequence_idx+1}/{len(sequences)}] {sequence}")
            parts = Path(sequence).stem.split('_')
            subject_name = Path(sequence).parent.stem

            with p.lock:
                if subject_name not in pose_sequences.keys():
                    pose_sequences[subject_name] = pymp.shared.dict()
                if subject_name not in charpose_indices.keys():
                    charpose_indices[subject_name] = pymp.shared.dict()
                if subject_name not in charposes.keys():
                    charposes[subject_name] = pymp.shared.dict()

            if len(parts[-1]) > 1 and not parts[-1] == 'Retake':
                parts.append('1')
            sample_id = '-'.join([subject_name] + parts)
            if sample_id in CONVERT_SAMPLE_ID:
                sample_id = CONVERT_SAMPLE_ID[sample_id]
            if sample_id is None:
                continue

            action_id = '-'.join(sample_id.split('-')[1:])

            # EXTRACT SKELETON FROM GRAB
            seq_data = grab_utils.parse_npz(sequence)
            n_comps = seq_data.n_comps
            gender = seq_data.gender
            T = seq_data.n_frames

            sbj_mesh = input_path / seq_data.body.vtemp[21:]
            sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)

            sbj_m = smplx.create(model_path=str(input_path),
                                 model_type='smplx',
                                 gender=gender,
                                 num_pca_comps=n_comps,
                                 v_template=sbj_vtemp,
                                 batch_size=T)

            sbj_parms = grab_utils.params2torch(seq_data.body.params)
            joints_sbj = grab_utils.to_cpu(sbj_m(**sbj_parms).joints)

            # SELECT FRAMES
            pickup_frame_idx = np.array([int(np.nonzero(np.sum(seq_data['contact']['object'], axis=1))[0][0])])[0]
            putdown_frame_idx = np.array([int(np.nonzero(np.sum(seq_data['contact']['object'], axis=1))[0][-1])])[0]

            action_frame_offset = annotations[sample_id]

            action_frame_idx = pickup_frame_idx + action_frame_offset

            pose_index_dict = {
                'initial_tpose': 0,
                'pickup': pickup_frame_idx,
                'action': action_frame_idx,
                'putdown': putdown_frame_idx,
                'final_tpose': -1
            }

            openpose_skeleton_sequence = joints_sbj[:, SKELETON_SMPLX2OPENPOSE_FINGERHAND, :]

            poses = np.stack([openpose_skeleton_sequence[idx] for idx in pose_index_dict.values()])
            pose_indices = np.stack(list(pose_index_dict.values()))

            with p.lock:
                pose_sequences[subject_name][action_id] = openpose_skeleton_sequence
                charposes[subject_name][action_id] = poses
                charpose_indices[subject_name][action_id] = pose_indices

    np.savez(str(output_path / 'data_3d_grab'),
             pose_sequences={key: dict(value) for key, value in pose_sequences.items()},
             charposes={key: dict(value) for key, value in charposes.items()},
             charpose_indices={key: dict(value) for key, value in charpose_indices.items()})


if __name__ == '__main__':
    # You can change your input and output paths but this configuration works without further modifications
    main(
        input_path=Path(__file__).parent / 'input',
        output_path=Path(__file__).parent / 'output'
    )
