"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np
import sys
sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    ### BEGIN YOUR CODE
    #NOTES:
    #1) rb = reads file only but in binary format
    #2) remember 8 bites (where a bite is either 0 or 1) makes up a byte. Thus, to get 32 bit
     #integer, we need to read 4 bits at a time.
    #3) "Endianness means that the bytes in computer memory are read in a certain order.
        #- BE (stores the big-end first)
        #- LE (stores little-end first)
        #ex) decimal number 9,499,938
            #* Big-Endian binary rep: 0b 10010000 11110101 00100010
            #* Little-Endian binary rep: 0b 00100010 11110101 10010000
       #Taken from:
       # https://www.freecodecamp.org/news/what-is-endianness-big-endian-vs-little-endian/

    #4) magic_number: A constant value used by lib magic (the tool which the file utility 
       #uses to guess file types). A magic number of 2051 tells the computer that the file type 
       #holds training data wheras 2049 tells the computer that the file is a label file

        #( num_images, rows and cols is standard info in the gzip files)
    #5) '>': format specifier indicates that the data should be interpreted in big-endian byte order. 
        #I: This format specifier specifies that we are expecting an unsigned integer that is 4 
        #   bytes in size (32 bits).
        #[0]: I: This format specifier specifies that we are expecting an unsigned integer that is 4 
             #bytes in size (32 bits).
         
    # Read and parse the header.
    with gzip.open(image_filesname, 'rb') as bin_image_file:
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

    return images, labels
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    # Calculate the softmax probabilities
    m = Z.shape[0]
    exp_Z = ndl.exp(Z)
    #exp_Z is 2 dim, but summation would return 1D, so we need to make summation 2D
    # based on how the code is written in the backward pass if you don't explicitley call broadcast
    # then it doesn't know that broadcast was used. 
    softmax_probs = exp_Z / ndl.summation(exp_Z, axes=(1,)).reshape((exp_Z.shape[0], 1)).broadcast_to((exp_Z.shape) )
    # Compute the loss for each sample in the batch
    log_t = ndl.log(softmax_probs) * y_one_hot
    loss = -ndl.summation(ndl.log(softmax_probs) * y_one_hot) / m
    # Average the loss over all samples
    return loss

    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr =0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X and y ARE STILL ARRAYS!!!!!
            -- X (np.ndarray[np.float32]): 2D input array of size
                (num_examples x input_dim).
            -- y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)

        #THESE ARE, HOWEVER, TENSORS!!!!
            -- W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
                (input_dim, hidden_dim)
            -- W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
                (hidden_dim, num_classes)
        
        This is just a float: 
            -- lr (float): step size (learning rate) for SGD
            batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    n = X.shape[0]
    m = W2.shape[1]
    y_one_hot = np.zeros((n, m))
    y_one_hot[np.arange(n), y] = 1
    # the number of steps that you need to take within
    # X. 
    steps = n // batch
    for i in range(steps + 1):
        start = i * batch
        # manually ensure that you do not go out of
        # bounds.
        end = min(start + batch, n)
        # we have reached end of X, so we are done
        if start >= end:
            break
        #create batches
        X_batch = ndl.Tensor(X[start: end])
        y_batch = ndl.Tensor(y_one_hot[start: end])
        #calculate the forward pass (logits)
        Z = ndl.matmul(ndl.relu(ndl.matmul(X_batch, W1)), W2)
        # loss
        loss = softmax_loss(Z, y_batch)
        # backward calls compute_gradient_of_variables which updates grad fields, 
        loss.backward()
        # grad
        W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
        W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())
    return (W1, W2)
    ### END YOUR SOLUTION

def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)