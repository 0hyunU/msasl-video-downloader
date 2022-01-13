import dataclasses
import numpy as np

@dataclasses
class Keypoint():
    # 3 dimensional ndarray data 
    # num_sequence(frame) * num_feature(body) * coordinate(x,y)
    point: np.ndarray


@dataclasses
class HolisticKeypoint():
    file_name: str
    action_class: str
    left_hand: Keypoint
    right_hand: Keypoint
    pose: Keypoint
    face: Keypoint