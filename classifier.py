import logging
import numpy as np
import cv2
import torch
from torch import tensor


class ShotAngleClassifier:
    def front_or_not(self,
                     masks: np.ndarray,
                     boxes: tensor,
                     logits: tensor,
                     angle_threshold: float,
                     circle_threshold: float
                     ):
        valid_idx = torch.where(logits > 0.5)[0]
        if len(valid_idx) < 2:
            return True
        else:
            boxes = boxes[valid_idx, :]
            masks = masks[valid_idx, :, :]
            angle, cs = self.calc_angle(masks, boxes)
            print('\nangle:', angle)

            if angle < angle_threshold:
                bool_x = self.circularity(cs[0], cs[1], circle_threshold)
                if bool_x:
                    '정면'
                else:
                    '정측면'
            else:
                return '측면'

    def calc_angle(self, masks, boxes):
        """
        TODO
        Args: TODO
        Returns: angle (0 - 90)
        """
        boxes = boxes.detach().cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        two_idx = sorted(range(len(areas)), key=lambda k: areas[k], reverse=True)[:2]

        mask_arr0 = masks[two_idx[0]]
        mask_arr1 = masks[two_idx[1]]

        contour0 = self.get_contour(mask_arr0)
        contour1 = self.get_contour(mask_arr1)
        two_contours = (contour0, contour1)

        cx0, cy0 = self.get_contour_center(contour0)
        cx1, cy1 = self.get_contour_center(contour1)

        return abs(np.arctan((cy0 - cy1) / (cx0 - cx1)) * 180 / np.pi), two_contours

    @staticmethod
    def get_contour(mask: np.ndarray):
        # mask image must be binary ([0, 255])
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours_sorted = sorted(contours, key=len, reverse=True)  # key=cv2.contourArea()

        if len(contours_sorted) > 1:
            first_contour, second_contour = contours_sorted[:2]
            #
            """
            아래 if문이 True 인 경우 설명                                        +------------------+
            => 이러한 경우는 바퀴가 잘려서 찍힌 경우, 마스크가 사변에 겹치는 경우       |XXXXXXXXXXXXXXXXXX|   
            => contour에 사변 중 일부가 포함되는 현상 발생                         +-------+XXXXXXXXXX|
                                                                              바퀴부분 |XXXXXXXXXX|
                                                                              +-------+XXXXXXXXXX|
                                                                              |XXXXXXXXXXXXXXXXXX|
                                                                              +------------------+            
            """
            if (first_contour[0, 0, 0] == 0) or (first_contour[0, 0, 1] == 0):
                return second_contour
            else:
                return first_contour
        else:
            # len(contours_sorted) == 1
            return contours_sorted[0]

    @staticmethod
    def get_contour_center(contour):
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return cx, cy
        else:
            logger = logging.getLogger()
            logger.info(f"바퀴의 중점을 찾을 수 없습니다.")

    @staticmethod
    def circularity(contour0, contour1, threshold):
        area = cv2.contourArea(contour0)           # 윤곽선의 면적
        perimeter = cv2.arcLength(contour0, True)  # 윤곽선의 둘레
        prob0 = 4 * np.pi * (area / (perimeter ** 2))

        area = cv2.contourArea(contour1)           # 윤곽선의 면적
        perimeter = cv2.arcLength(contour1, True)  # 윤곽선의 둘레
        prob1 = 4 * np.pi * (area / (perimeter ** 2))

        return (prob0 + prob1) / 2 < threshold
