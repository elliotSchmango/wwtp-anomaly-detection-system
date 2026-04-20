import cv2
import numpy as np
import logging
from config.settings import settings

logger = logging.getLogger("Eagle.Alignment")

class FrameStabilizer:
    @staticmethod
    def stabilize(current_frame, ref_frame):
        """
        Aligns current_frame to ref_frame using phase correlation to fix PTZ drift,
        then center crops both frames to remove border artifacts.
        """
        if current_frame is None or ref_frame is None:
            return current_frame, ref_frame

        if current_frame.shape != ref_frame.shape:
            return current_frame, ref_frame

        h, w = current_frame.shape[:2]
        
        # calculate cropping margins
        margin_y = int(h * settings.CROP_MARGIN_PERCENT)
        margin_x = int(w * settings.CROP_MARGIN_PERCENT)
        
        # if alignment is disabled, just do the center crop
        if not settings.ALIGNMENT_ENABLED:
            cropped_current = current_frame[margin_y:h-margin_y, margin_x:w-margin_x]
            cropped_ref = ref_frame[margin_y:h-margin_y, margin_x:w-margin_x]
            return cropped_current, cropped_ref

        # alignment enabled: use phase correlation
        # convert to grayscale float32
        gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_ref = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # find translation phase shift
        shift, _ = cv2.phaseCorrelate(gray_curr, gray_ref)
        shift_x, shift_y = shift

        # create a transformation matrix
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        
        # warp the current frame to align with the reference
        aligned_curr = cv2.warpAffine(current_frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # now crop out the borders since those pixels might be missing/replicated from the shift
        cropped_curr = aligned_curr[margin_y:h-margin_y, margin_x:w-margin_x]
        cropped_ref = ref_frame[margin_y:h-margin_y, margin_x:w-margin_x]

        return cropped_curr, cropped_ref
