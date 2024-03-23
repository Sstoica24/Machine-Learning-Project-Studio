import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            # ::-1 is used to reverse the order in a specific axis/dim. We want to flip horizontally
            # therefore, we reverse 1st dim (playing around showed this).
            # Rather than providing a "beginning" and an "end" index,
            # it's telling Python to skip by every -1 objects in the array. It's effectively
            # reversing the array.
            return img[:, ::-1, :]
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        # we have a list because image is 3 dimensions. For the first 2 enteries associated with 1th
        # and 1st 2nd dim, we want want (self.padding, self.padding) as it gives the desired padding for the 
        # axes. We need to leave 3rd dim alone ==> we use (0,0).
        img_pad = np.pad(img, [(self.padding, self.padding), (self.padding, self.padding), (0, 0)], 'constant')
        H, W, _ = img_pad.shape
        # we want to crop the image and do it within bounds (not where we have padded the image), thus, we use
        # self.padding (jump past section which is padding) + [stift_x or shift_y] : dimension_shape - self.padding (skip area which is padded) + 
        # [shift_x or shift_y]. The reason we add shift_x or shift_y is because we need to shift image on both ends to 
        # successfuly crop.
        return img_pad[self.padding + shift_x: H - self.padding + shift_x, self.padding + shift_y: W - self.padding + shift_y, :]
        ### END YOUR SOLUTION
