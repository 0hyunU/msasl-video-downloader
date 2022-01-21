import dataclasses
import numpy as np
#import tensorflow as tf

@dataclasses.dataclass
class HolisticKeypoint():
    # file information
    file_path: str
    file_name: str
    action_class: str
    avail_frame_len: int

    # 3 dimensional ndarray data 
    # num_sequence(frame) * num_feature(body) * coordinate(x,y)
    left_hand: np.ndarray
    right_hand: np.ndarray
    pose: np.ndarray
    face: np.ndarray
    
    def get_tensor(self) -> np.ndarray:
        res = np.hstack([self.left_hand, self.right_hand,self.pose,self.face])
        return res #tf.convert_to_tensor(tmp)

    @property
    def get_pose_data_len(self) -> int:
        return len(self.pose)
    
    @property
    def get_face_data_len(self) -> int:
        return len(self.face)
    
    @property
    def get_lh_data_len(self) -> int:
        return len(self.left_hand)
    @property
    def get_rh_data_len(self) -> int:
        return len(self.right_hand)