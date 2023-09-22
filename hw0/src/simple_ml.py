import struct #python modul used to handle binary data stored in files
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass

from simple_ml_cython import *


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
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
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
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

    return images, labels
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    #softamax is e^Z/sum(e^Z)
    #The softmax loss for a single example is computed as the negative log 
    # of the predicted probability assigned to the correct class.
    #subtracting max of Z is a stability technique to prevent large exponentials that
    # could lead to overflow issues.
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True)) #max over each col ==> over class
    softmax_scores = exp_Z / np.sum(exp_Z, axis=1, keepdims =True)
    batch_size = Z.shape[0]
    loss = -np.log(softmax_scores[np.arange(batch_size), y])
    return np.mean(loss)
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).

        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)

        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)

        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_examples = X.shape[0]
    num_classes = theta.shape[1]

    #we want to index 0, batch, 2*batch, 3*batch, ...,
    # so that we can split X and y into the necessary batch sizes
    for i in range(0, num_examples, batch):
        X_batch = X[i:i+batch]
        y_batch = y[i:i+batch]

        #gradient of softmax = 1/batch * X^T(softmax(hypothesis function) - one_hot_encoded(y))
        #as given in the notebook
        hypothesis = np.dot(X_batch, theta)
        exp_hypothesis = np.exp(hypothesis - np.max(hypothesis, axis=1, keepdims=True))
        softmax_probs = exp_hypothesis / np.sum(exp_hypothesis, axis=1, keepdims=True)

        #np.eye(num_classes)[y_batch] is a beautiful and slick way to get the one_hot_encodings of y.
        #it tells python use the elements of y to index the ones in the numpy array. 
        grad = 1/batch * np.dot(X_batch.T, softmax_probs - np.eye(num_classes)[y_batch])

        theta -= lr * grad

    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).

        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)

        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)

        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)

        lr (float): step size (learning rate) for SGD
        
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_examples = X.shape[0]
    num_classes = W2.shape[1]

    #same idea: loop over all examples, but indexing only at 0, batch, 2*batch...
    #to create the batches for X and y
    for i in range(0, num_examples, batch):
        # Select a minibatch
        X_batch = X[i:i+batch]
        y_batch = y[i:i+batch]

        # Forward pass
        Z = np.dot(X_batch, W1)
        Z_relu = np.maximum(0, Z)  # ReLU activation

        Z_out = np.dot(Z_relu, W2)

        # = = = = = Calculating gradients as given in notebook = = = = = #
        #note: grad for W2 is identitical to softmax regression case
        exp_Z_out = np.exp(Z_out - np.max(Z_out, axis=1, keepdims=True))
        softmax_probs = exp_Z_out / np.sum(exp_Z_out, axis=1, keepdims=True)
        G2 = softmax_probs - np.eye(num_classes)[y_batch]
        W2_grad = 1/batch * np.dot(Z_relu.T, G2)

        #compute grad for W1_grad
        G1 = np.dot(G2, W2.T) * (Z > 0).astype(int)  # ReLU derivative
        W1_grad = 1/batch * np.dot(X_batch.T, G1)

        # Update weights
        W1 -= lr * W1_grad 
        W2 -= lr * W2_grad
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


#this will be used for test code when just running the file. However, 
#if you import this file, since it is labeled as "main", you won't
#be able to touch this code.
if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
