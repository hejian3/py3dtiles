import numpy as np


def make_rotation_matrix(z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    v0 = z1 / np.linalg.norm(z1)
    v1 = z2 / np.linalg.norm(z2)

    angle = np.arccos(np.clip(np.dot(v0, v1), -1.0, 1.0))
    direction = np.cross(v0, v1)

    sina = np.sin(angle)
    cosa = np.cos(angle)
    direction[:3] /= np.sqrt(np.dot(direction[:3], direction[:3]))
    # rotation matrix around unit vector
    rotation_matrix = np.diag([cosa, cosa, cosa])
    rotation_matrix += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    rotation_matrix += np.array(
        [
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ]
    )
    final_rotation_matrix = np.identity(4)
    final_rotation_matrix[:3, :3] = rotation_matrix

    return final_rotation_matrix

def make_scale_matrix(factor: float) -> np.ndarray:
    scale_matrix = np.diag([factor, factor, factor, 1.0])
    return scale_matrix

def make_translation_matrix(direction: np.ndarray) -> np.ndarray:
    translation_matrix = np.identity(4)
    translation_matrix[:3, 3] = direction[:3]
    return translation_matrix
