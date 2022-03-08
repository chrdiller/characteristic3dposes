import numpy as np

RED = (0, 1, 1)
ORANGE = (20/360, 1, 1)
YELLOW = (60/360, 1, 1)
GREEN = (100/360, 1, 1)
CYAN = (175/360, 1, 1)
BLUE = (210/360, 1, 1)

RED_DARKER = (0, 1, 0.25)
ORANGE_DARKER = (20/360, 1, 0.25)
YELLOW_DARKER = (60/360, 1, 0.25)
GREEN_DARKER = (100/360, 1, 0.25)
CYAN_DARKER = (175/360, 1, 0.25)
BLUE_DARKER = (210/360, 1, 0.25)

ACTIONS = ['SittingDown', 'Posing', 'Greeting', 'Directions', 'Waiting', 'WalkDog', 'Purchases', 'WalkTogether', 'Sitting', 'Eating', 'Smoking', 'TakingPhoto', 'Phoning', 'WalkingDog', 'Discussion', 'Photo', 'Walking']

PHASE_SUBJECTS = {'train': ['S1', 'S6', 'S7', 'S8', 'S9'], 'val': ['S11'], 'test': ['S5']}


class Skeleton17:
    """
    Reduced skeleton, usually used for motion prediction.
    It removes all duplicate joints as well as hand and feet joints
    """
    num_joints = 17

    head = [9, 10]
    right_arm = [14, 15, 16]
    left_arm = [11, 12, 13]
    rh = [16]
    lh = [13]
    right_leg = [1, 2, 3]
    left_leg = [4, 5, 6]
    spine = [0, 7, 8]
    hip = [0]

    end_effectors = rh + lh

    bones =       [(0, 1),      (1, 2),      (2, 3),      (0, 4),      (4, 5),      (5, 6),      (0, 7),        (7, 8),        (8, 9),       (9, 10),      (9, 11),       (11, 12),      (12, 13),      (9, 14),    (14, 15),   (15, 16)]
    bone_colors = [BLUE_DARKER, BLUE_DARKER, BLUE_DARKER, CYAN_DARKER, CYAN_DARKER, CYAN_DARKER, YELLOW_DARKER, YELLOW_DARKER, GREEN_DARKER, GREEN_DARKER, ORANGE_DARKER, ORANGE_DARKER, ORANGE_DARKER, RED_DARKER, RED_DARKER, RED_DARKER]

    joint_names = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
    joint_sizes = [0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.048, 0.032, 0.032, 0.048]
    joint_colors = [YELLOW, BLUE, BLUE, BLUE, CYAN, CYAN, CYAN, YELLOW, YELLOW, GREEN, GREEN, ORANGE, ORANGE, ORANGE, RED, RED, RED]

    angle_joints = [
        [0, 1, 2],  # Right Hip
        [0, 4, 5],  # Left Hip

        [1, 2, 3],  # Right Leg
        [4, 5, 6],  # Left Leg

        [0, 7, 8],  # Spine
        [7, 8, 9],  # Neck
        [8, 9, 10],  # Head

        [9, 11, 12],  # Left Shoulder
        [11, 12, 13],  # Left Ellbow

        [9, 14, 15],  # Right Shoulder
        [14, 15, 16],  # Right Ellbow
    ]

    angle_weights = [
        1., 1.,  # Hip
        1., 1.,  # Legs
        5., 5., 5.,  # Spine
        1., 1.,  # Left Arm
        1., 1.,  # Right Arm
    ]

    joints_weights = [
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
    ]

    @staticmethod
    def from_32(skeleton: np.array) -> np.array:
        return skeleton[:, [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]]


class Skeleton32:
    """
    The full skeleton, as recorded.
    Contains some duplicate joints: 0==11(hip), 13==16==24(Neck), 19==20(Left Hand), 22==23(Left Finger), 27==28(Right Hand), 30==31(Right Finger)
    """
    num_joints = 32

    head = [14, 15]
    right_arm = [25, 26, 27, 28, 29, 30, 31]
    left_arm = [17, 18, 19, 20, 21, 22, 23]
    rh = [27, 28, 29, 30, 31]
    lh = [19, 20, 21, 22, 23]
    right_leg = [1, 2, 3, 4, 5]
    left_leg = [6, 7, 8, 9, 10]
    spine = [0, 11, 12, 13, 16, 24]
    hip = [0, 11]

    bones = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 6), (6, 7), (7, 8), (8, 9), (9, 10), (0, 11), (11, 12), (12, 13), (13, 14), (14, 15), (12, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (19, 22), (22, 23), (24, 25), (13, 25), (25, 26), (26, 27), (27, 28), (27, 29), (27, 30)]

    angle_joints = [
        [0, 1, 2],  # Right Hip
        [11, 1, 2],  # Right Hip
        [0, 6, 7],  # Left Hip
        [11, 6, 7],  # Left Hip

        [3, 4, 5],  # Right Foot
        [2, 3, 4],  # Right Foot (Heel)
        [7, 8, 9],  # Left Foot (Heel)
        [8, 9, 10],  # Left Foot

        [0, 12, 24],  # Spine
        [11, 12, 24],  # Spine

        [12, 13, 14],  # Neck
        [12, 16, 14],  # Neck
        [12, 24, 14],  # Neck
        [13, 14, 15],  # Head
        [16, 14, 15],  # Head
        [24, 14, 15],  # Head

        [16, 17, 18],  # Left Shoulder
        [13, 17, 18],  # Left Shoulder
        [24, 17, 18],  # Left Shoulder
        [17, 18, 19],  # Left Ellbow
        [17, 18, 20],  # Left Ellbow
        [18, 19, 21],  # Left Hand
        [18, 20, 21],  # Left Hand
        [18, 19, 22],  # Left Hand
        [18, 19, 23],  # Left Hand
        [18, 20, 22],  # Left Hand
        [18, 20, 23],  # Left Hand
        [21, 19, 22],  # Left Hand
        [21, 19, 23],  # Left Hand
        [21, 20, 22],  # Left Hand
        [21, 20, 23],  # Left Hand

        [16, 25, 26],  # Right Shoulder
        [13, 25, 26],  # Right Shoulder
        [24, 25, 26],  # Right Shoulder
        [25, 26, 27],  # Right Ellbow
        [25, 26, 28],  # Right Ellbow
        [26, 27, 29],  # Right Hand
        [26, 28, 29],  # Right Hand
        [26, 27, 30],  # Right Hand
        [26, 27, 31],  # Right Hand
        [26, 28, 30],  # Right Hand
        [26, 28, 31],  # Right Hand
        [30, 27, 29],  # Right Hand
        [30, 28, 29],  # Right Hand
        [31, 27, 29],  # Right Hand
        [31, 28, 29],  # Right Hand
    ]

    angle_weights = [
        1., 1., 1., 1.,  # Hip
        1., 1., 1., 1.,  # Feet
        5., 5.,  # Spine
        1., 1., 1., 1., 1., 1.,  # Head
        1., 1., 1., 1., 1., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,  # Left Arm
        1., 1., 1., 1., 1., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,  # Right Arm
    ]

    JOINT_WEIGHTS = [
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    ]
