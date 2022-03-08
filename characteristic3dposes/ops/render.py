from pathlib import Path

import numpy as np
from trimesh import creation
import pyrender
from pyrender.constants import RenderFlags
import cv2

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'


def hsv_to_rgb(h, s, v):
    """
    Easy-to-use HSV to RGB color space transformation

    :param h: Hue
    :param s: Saturation
    :param v: Value
    :return: The RGB equivalent
    """
    shape = (1,)
    i = np.int_(h*6.)
    f = h*6.-i

    q = f
    t = 1.-f
    i = np.ravel(i)
    f = np.ravel(f)
    i%=6

    t = np.ravel(t)
    q = np.ravel(q)

    clist = (1-s*np.vstack([np.zeros_like(f), np.ones_like(f),q,t]))*v

    #0:v 1:p 2:q 3:t
    order = np.array([[0,3,1],[2,0,1],[1,0,3],[1,2,0],[3,1,0],[0,1,2]])
    rgb = clist[order[i], np.arange(np.prod(shape))[:,None]]

    return rgb.reshape(shape+(3,))


def rotation_matrix_from_vectors(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2

    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if c == -1:
        return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    if c == 1:
        return np.eye(3)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    return rotation_matrix


def render_skeleton(skeleton: np.array, skeleton_cls, out_path: Path, resolution=720):
    """
    Render a skeleton, defined by its xyz coordinates in a nx3 numpt array and corresponding properties in skeleton_cls

    :param skeleton: A nx3 numpy array, containing the skeleton joints' xyz locations in 3d space
    :param skeleton_cls: The skeleton class, containing at least the properties joint_sizes, joint_colors, bones
    :param out_path: The path to save the final rendered image to
    :param resolution: The resolution == number of pixels along x and y (rendered image will be quadratic)
    :return:
    """
    # Renderer and Scene
    r = pyrender.OffscreenRenderer(viewport_width=resolution, viewport_height=resolution, point_size=1.0)
    scene = pyrender.Scene(bg_color=(255, 255, 255, 0))
    # Camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.5000000, -0.8660254, -2.5],
        [0.0, 0.8660254,  0.5000000, 2.75],
        [0.0, 0.0, 0.0, 1.0],
        ])
    scene.add(camera, pose=camera_pose)
    # Light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.)
    scene.add(light, pose=camera_pose)

    # Joints
    for joint_idx, joint in enumerate(skeleton):
        mesh = creation.icosphere(radius=skeleton_cls.joint_sizes[joint_idx])
        mesh.visual.vertex_colors = tuple(np.int_(hsv_to_rgb(*skeleton_cls.joint_colors[joint_idx]) * 255.)[0])[::-1] + (255,)
        mesh_pose = np.eye(4)
        mesh_pose[:3, 3] = joint
        scene.add(pyrender.Mesh.from_trimesh(mesh), pose=mesh_pose)

    # Bones
    for bone_idx, bone in enumerate(skeleton_cls.bones):
        bone_vector = skeleton[bone[1]] - skeleton[bone[0]]
        bone_length = np.linalg.norm(bone_vector)
        if np.linalg.norm(bone_vector) == 0:
            continue
        mesh = creation.cylinder(radius=0.01, height=bone_length)
        mesh_pose = np.eye(4)
        mesh_pose[:3, :3] = rotation_matrix_from_vectors([0, 0, bone_length], bone_vector)
        mesh_pose[:3, 3] = skeleton[bone[0]] + mesh_pose[:3, :3] @ np.array([0, 0, bone_length / 2.])
        scene.add(pyrender.Mesh.from_trimesh(mesh), pose=mesh_pose)

    # Render
    color, _ = r.render(scene, flags=RenderFlags.SHADOWS_ALL | RenderFlags.RGBA)

    # Write image
    cv2.imwrite(str(out_path), color)


if __name__ == '__main__':
    raise NotImplementedError
