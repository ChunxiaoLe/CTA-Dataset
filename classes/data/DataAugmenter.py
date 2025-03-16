import math
import random
from typing import Tuple

import cv2
import numpy as np

from auxiliary.utils import rgb_to_bgr


class DataAugmenterseq():

    def __init__(self, input_size: Tuple = (224, 224)):
        self.__input_size = input_size
        #角度15，scale0.8-1.0, color0.0
        self.__angle = 15
        self.__scale = [0.8, 1.0]
        self.__color = 0.0

    @staticmethod
    def __rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotates an OpenCV 2 / NumPy image about it's centre by the given angle (in degrees).
        The returned image will be large enough to hold the entire new image, with a black background
        """

        # Get the image size (note: NumPy stores image matrices backwards)
        image_size = (image.shape[1], image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])

        rot_mat_no_translate = np.matrix(rot_mat[0:2, 0:2])

        image_w2, image_h2 = image_size[0] * 0.5, image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2, image_h2]) * rot_mat_no_translate).A[0],
            (np.array([image_w2, image_h2]) * rot_mat_no_translate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_no_translate).A[0],
            (np.array([image_w2, -image_h2]) * rot_mat_no_translate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos, x_neg = [x for x in x_coords if x > 0], [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos, y_neg = [y for y in y_coords if y > 0], [y for y in y_coords if y < 0]

        right_bound, left_bound, top_bound, bot_bound = max(x_pos), min(x_neg), max(y_pos), min(y_neg)
        new_w, new_h = int(abs(right_bound - left_bound)), int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([[1, 0, int(new_w * 0.5 - image_w2)], [0, 1, int(new_h * 0.5 - image_h2)], [0, 0, 1]])

        # Compute the transform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        return cv2.warpAffine(image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)

    @staticmethod
    def __largest_rotated_rect(w: float, h: float, angle: float) -> Tuple:
        """
        Given a rectangle of size w x h that has been rotated by 'angle' (in radians), computes the width and height of
        the largest possible axis-aligned rectangle within the rotated rectangle.

        Original JS code by 'Andri' and Magnus Hoff from Stack Overflow. Converted to Python by Aaron Snoswell
        """
        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        length = h if (w < h) else w
        d = length * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)
        delta = math.pi - alpha - gamma
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return bb_w - 2 * x, bb_h - 2 * y

    @staticmethod
    def __crop_around_center(image: np.ndarray, width: float, height: float) -> np.ndarray:
        """ Given a NumPy / OpenCV 2 image, crops it to the given width and height around it's centre point """

        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        width = image_size[0] if width > image_size[0] else width
        height = image_size[1] if height > image_size[1] else height

        x1, x2 = int(image_center[0] - width * 0.5), int(image_center[0] + width * 0.5)
        y1, y2 = int(image_center[1] - height * 0.5), int(image_center[1] + height * 0.5)

        return image[y1:y2, x1:x2]

    def __rotate_and_crop(self, image: np.ndarray, angle: float) -> np.ndarray:
        if angle is None:
            angle = self.__get_random_angle()
        width, height = image.shape[:2]
        target_width, target_height = self.__largest_rotated_rect(width, height, math.radians(angle))
        return self.__crop_around_center(self.__rotate_image(image, angle), target_width, target_height)

    def __rescale(self, img: np.ndarray, scale: float) -> np.ndarray:
        if scale is None:
            scale = self.__get_random_scale(img)
        start_x = random.randrange(0, img.shape[0] - scale + 1)
        start_y = random.randrange(0, img.shape[1] - scale + 1)
        return img[start_x:start_x + scale, start_y:start_y + scale]

    def resize(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, self.__input_size)

    def resize_sequence(self, img: np.ndarray) -> np.ndarray:
        return np.stack([self.resize(img[i]) for i in range(img.shape[0])])

    def __get_random_scale(self, img: np.ndarray) -> float:
        scale = math.exp(random.random() * math.log(self.__scale[1] / self.__scale[0])) * self.__scale[0]
        return min(max(int(round(min(img.shape[:2]) * scale)), 10), min(img.shape[:2]))

    def __get_random_angle(self) -> float:
        return (random.random() - 0.5) * self.__angle

    def get_random_color_bias(self):
        color_bias = np.zeros(shape=(3, 3))
        for i in range(3):
            color_bias[i, i] = 1 + random.random() * self.__color - 0.5 * self.__color
        return color_bias

    @staticmethod
    def __random_flip(img: np.ndarray, p: int = None) -> np.ndarray:
        """
        Perform random left/right flip with probability p (defaults to 0.5)
        @param img: the image to be flipped
        @param p: the probability according with image should be flipped (in {0, 1})
        @return: the flipped image with probability p, else the original image
        """
        if p is None:
            p = random.randint(0, 1)
        return img[:, ::-1].astype(np.float32) if p else img.astype(np.float32)

    def __augment_image(self,
                        img: np.ndarray,
                        color_bias: np.ndarray = None,
                        scale: float = None,
                        angle: float = None,
                        flip_p: int = None) -> np.ndarray:
        img = self.__random_flip(self.resize(self.__rotate_and_crop(self.__rescale(img, scale), angle)), flip_p)
        return np.clip(self.__apply_color_bias(img, color_bias), 0, 255)
        #return img

    def __augment_illuminant(self, illuminant: np.ndarray, color_bias: np.ndarray = None) -> np.ndarray:
        if color_bias is None:
            color_bias = self.get_random_color_bias()
        illuminant = rgb_to_bgr(illuminant)
        new_illuminant = np.array([[illuminant[j] * color_bias[i, j] for j in range(3)] for i in range(3)])
        return rgb_to_bgr(np.clip(new_illuminant, 0.01, 100))
        #return new_illuminant

    def __apply_color_bias(self, img: np.ndarray, color_bias: np.ndarray) -> np.ndarray:
        if color_bias is None:
            color_bias = self.get_random_color_bias()
        return img * np.array([[[color_bias[0][0], color_bias[1][1], color_bias[2][2]]]], dtype=np.float32)

    def augment_sequence(self, images: np.ndarray, illuminant: np.ndarray) -> Tuple:
        color_bias = self.get_random_color_bias()
        scale = self.__get_random_scale(images[0])
        angle = self.__get_random_angle()
        #flip_p = random.randint(0, 1)
        flip_p = 0

        augmented_frames, augmented_illuminants = [], []
        for i in range(images.shape[0]):
            augmented_frames.append(self.__augment_image(images[i], color_bias, scale, angle, flip_p))
            augmented_illuminants.append(self.__augment_illuminant(illuminant, color_bias))

        return np.stack(augmented_frames), np.mean(augmented_illuminants, axis=0)

    def augment_mimic(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 4:
            num_steps = img.shape[0]
            img = img[-1]
        elif len(img.shape) == 3:
            num_steps = 1
            img = img
        else:
            raise ValueError("Bad image shape detected augmenting mimic: {}".format(img.shape))

        augmented_frames, img_temp = [], img[:, :, ::-1] * (1.0 / 255)
        for _ in range(num_steps):
            angle = self.__get_random_angle()
            scale = min(max(int(round(min(img_temp.shape[:2]) * 0.95)), 10), min(img_temp.shape[:2]))
            img_temp = self.resize(self.__rotate_and_crop(self.__rescale(img_temp, scale), angle))
            img_temp = np.clip(img_temp.astype(np.float32), 0, 255)
            augmented_frames.append(img_temp)

        return np.stack(augmented_frames)
