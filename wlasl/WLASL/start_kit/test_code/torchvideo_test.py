import torchvideo.transforms as VT
import torchvision.transforms as IT
from torchvision.transforms import Compose

transform = Compose([
    VT.CenterCropVideo((224, 224)),  # (h, w)
    VT.CollectFrames(),
    VT.PILVideoToTensor(),
    VT.TimeToChannel()
])