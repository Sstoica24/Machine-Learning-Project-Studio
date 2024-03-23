from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip
import needle as ndl

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename, 'rb') as bin_image_file:
            magic_number = struct.unpack('>I', bin_image_file.read(4))[0]
            num_images = struct.unpack('>I', bin_image_file.read(4))[0]
            num_rows = struct.unpack('>I', bin_image_file.read(4))[0]
            num_columns = struct.unpack('>I', bin_image_file.read(4))[0]

            # Read binary image data.
            image_data = bin_image_file.read() #reads all the binary data from the current position 
            #of image_file to the end of the file and stores it in the image_data variable. 
            #every bit from now on represents a pixel bit. 
    
        # Read label file.
        with gzip.open(label_filename, 'rb') as label_file:
            # Read and parse the header.
            magic_number_labels = struct.unpack('>I', label_file.read(4))[0]
            num_labels = struct.unpack('>I', label_file.read(4))[0]

            # Read binary label data.
            label_data = label_file.read()
        
        # Ensure header information matches.
        #magic_number = 2051 means we are working with images
        #magic_number_labels = 2049 means that we are working with labels
        if magic_number != 2051 or magic_number_labels != 2049 or num_images != num_labels:
            raise ValueError("Invalid MNIST file format.")

        #buffers: a reserved segment of memory (RAM) within a program that is used to hold the 
        # data being processed.
        #therefore, because image_data and label_data is just data we are working with buffers. 
        #np.frombuffer convert info in buffer into a 1D nump array
        images = np.frombuffer(image_data, dtype=np.uint8, offset=0)
        #reshape the array to needed dimensions (num_images, num_rows*num_cols)
        images = images.reshape(num_images, num_rows*num_columns)
        #normalize
        images = images.astype(np.float32) / 255.0
        labels = np.frombuffer(label_data, dtype=np.uint8, offset=0)

        # now properly initialize them
        self.images = images
        self.labels = labels
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        # can take batches too (which is what they don't tell you in the statment)
        # since index can be a range
        imgs = self.images[index]
        labels = self.labels[index]
        # greater than 1 dimension and is a batch (represented when imgs.shape[0] > 1)
        # testing showed that we need to flatten array.
        if len(imgs.shape) > 1 and imgs.shape[0] > 1:
            # Need call Flatten class we implemented rather than flatten() from numpy
            # because we need our array to be (batch_size, dim_1 * dim_2 *...* dim_n).
            # note, transform functions expect image of dim (H, W , C)
            X = ndl.Tensor([self.apply_transforms(img.reshape(28,28,1)) for img in imgs])
            # how they did it when they tested if Flatten was implemented correctly
            tform = ndl.nn.Flatten()
            images = tform(X).cached_data
        else:
            images = np.array(self.apply_transforms(imgs.reshape(28, 28, 1)).flatten())
        return (images, labels)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION