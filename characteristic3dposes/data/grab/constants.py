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

ACTIONS = ['call', 'drink', 'shake', 'squeeze', 'brush', 'clean', 'play', 'use', 'toast', 'set', 'stamp', 'staple', 'screw', 'pour', 'peel', 'see', 'browse', 'cook', 'inspect', 'lift', 'eat', 'open', 'on', 'offhand', 'takepicture', 'chop', 'fly', 'wear', 'pass']
OBJECTS = ['train', 'pyramidsmall', 'toruslarge', 'duck', 'spherelarge', 'torusmedium', 'flashlight', 'knife', 'stapler', 'mouse', 'spheresmall', 'fryingpan', 'banana', 'phone', 'hammer', 'toothpaste', 'scissors', 'pyramidlarge', 'cubelarge', 'airplane', 'alarmclock', 'cylinderlarge', 'cylindermedium', 'cubemedium', 'spheremedium', 'toothbrush', 'doorknob', 'headphones', 'teapot', 'hand', 'waterbottle', 'pyramidmedium', 'mug', 'flute', 'watch', 'piggybank', 'gamecontroller', 'wineglass', 'cubesmall', 'camera', 'cup', 'lightbulb', 'bowl', 'cylindersmall', 'apple', 'stanfordbunny', 'torussmall', 'stamp', 'binoculars', 'elephant', 'eyeglasses']
SUBJECTS = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']

PHASE_SUBJECTS = {'train': ['s2', 's3', 's4', 's5', 's6', 's7', 's8', 's9'], 'val': ['s1'], 'test': ['s10']}


class Skeleton25:
    num_joints = 25

    head = [0, 15, 16, 17, 18]
    right_arm = [2, 3, 4]
    left_arm = [5, 6, 7]
    right_hand = [4]
    left_hand = [7]
    right_leg = [9, 10, 11, 22, 23, 24]
    left_leg = [12, 13, 14, 19, 20, 21]
    spine = [1, 8]
    hip = [8]

    end_effectors = right_hand + left_hand

    joint_names = ['nose', 'neck', 'r-shoulder', 'r-ellbow', 'r-hand', 'l-shoulder', 'l-ellbow', 'l-hand', 'mid-hip', 'r-hip', 'r-knee', 'r-ankle', 'l-hip', 'l-knee', 'l-ankle', 'r-eye', 'l-eye', 'r-ear', 'l-ear', 'l-bigtoe', 'l-smalltoe', 'l-heel', 'r-bigtoe', 'r-smalltoe', 'r-heel']
    joint_colors = [GREEN, YELLOW, RED, RED, RED, ORANGE, ORANGE, ORANGE, YELLOW, BLUE, BLUE, BLUE, CYAN, CYAN, CYAN, GREEN, GREEN, GREEN, GREEN, CYAN, CYAN, CYAN, BLUE, BLUE, BLUE]
    joint_sizes = [0.032, 0.032, 0.032, 0.032, 0.048, 0.032, 0.032, 0.048, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032]

    bones = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (0, 15), (15, 17), (0, 16), (16, 18), (1, 8), (8, 9), (9, 10), (10, 11), (11, 24), (11, 22), (22, 23), (8, 12), (12, 13), (13, 14), (14, 21), (14, 19), (19, 20)]
    bone_colors = [GREEN_DARKER, RED_DARKER, RED_DARKER, RED_DARKER, ORANGE_DARKER, ORANGE_DARKER, ORANGE_DARKER, GREEN_DARKER, GREEN_DARKER, GREEN_DARKER, GREEN_DARKER, YELLOW_DARKER, BLUE_DARKER, BLUE_DARKER, BLUE_DARKER, BLUE_DARKER, BLUE_DARKER, BLUE_DARKER, BLUE_DARKER, BLUE_DARKER, BLUE_DARKER, BLUE_DARKER, BLUE_DARKER, BLUE_DARKER]

    angle_joints = [
        [2, 1, 5],
        [8, 1, 0],
        [1, 0, 15],
        [1, 0, 16],
        [15, 0, 16],
        [0, 15, 17],
        [0, 16, 18],
        [9, 8, 12],
        [9, 8, 1],
        [12, 8, 1],
        [10, 9, 8],
        [11, 10, 9],
        [22, 11, 10],
        [24, 11, 10],
        [23, 22, 11],
        [8, 12, 13],
        [12, 13, 14],
        [13, 14, 19],
        [13, 14, 21],
        [14, 19, 20]
    ]

    angle_weights = [
        1., 1., 10., 10., 10., 10., 10., 1., 1., 1., 1., 1., 5., 5., 5., 1., 1., 5., 5., 5.,
    ]

    joints_weights = [
        .1, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., .1, 1., 1., .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1
    ]
