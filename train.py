import os
import urllib.request
import gzip
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import pickle
from cnn import Full_Net, Relu, Softmax, SGD 

class ModelTrainer:
    def __init__(self, model) -> None:
        """
        Initialize ModelTrainer.

        Args:
            model: The model to train.
        """
        self.model = model

    def train_model(self, train_data, train_target, test_data, test_target, epochs, batch_size):
        """
        Train the model on a dataset.

        Args:
            train_data (np.ndarray): Training input data.
            train_target (np.ndarray): Training target data.
            test_data (np.ndarray): Test input data.
            test_target (np.ndarray): Test target data.
            epochs (int): Number of epochs to train for.
            batch_size (int): Size of each batch.
        """
        plt.ion()
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        ax_loss, ax_accuracy = axs
        self.epoch_counter = []  # Store epoch numbers

        self.train_loss_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []
        self.train_accuracy_history = []

        pbar = tqdm(range(epochs))
        for epoch in pbar:
            train_data, train_target = shuffle_data(train_data, train_target)
            train_accuracies = []
            train_losses = []
            for i in range(0, train_data.shape[0], batch_size):
                data_batch = train_data[i:i + batch_size]
                target_batch = train_target[i:i + batch_size]
                output = self.model.forward(data_batch)
                loss = self.model.backward(output, target_batch)
                acc = compute_accuracy(output, target_batch)

                train_losses.append(loss)
                train_accuracies.append(acc)
            
            train_loss = np.mean(train_losses)
            test_output = self.model.forward(test_data)
            test_loss, _ = self.model.loss_object.forward(test_output, test_target)
            test_accuracy = compute_accuracy(test_output, test_target)
            train_accuracy = np.mean(train_accuracies)
        
            self.train_loss_history.append(train_loss)
            self.test_loss_history.append(test_loss)
            self.test_accuracy_history.append(test_accuracy)
            self.train_accuracy_history.append(train_accuracy)
            
            self.epoch_counter.append(epoch)  # Add epoch number
            update_progress_bar(pbar, epoch + 1, train_loss, test_loss, test_accuracy)
            
            #update training visualization
            self.update_training_visualization(ax_loss, ax_accuracy)

            fig.canvas.draw()
            plt.pause(0.01)
    
        # Turn off interactive mode after training
        plt.ioff()
        plt.show()
        return model

    def update_training_visualization(self, ax_loss, ax_accuracy):
        """
        Visualize the training process.
        """
        ax_loss.clear()
        ax_loss.plot(self.epoch_counter, self.train_loss_history, label='Train Loss')
        ax_loss.plot(self.epoch_counter, self.test_loss_history, label='Test Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()

        
        ax_accuracy.clear()
        ax_accuracy.plot(self.epoch_counter, self.train_accuracy_history, label='Train Accuracy')
        ax_accuracy.plot(self.epoch_counter, self.test_accuracy_history, label='Test Accuracy')
        ax_accuracy.set_xlabel('Epoch')
        ax_accuracy.set_ylabel('Accuracy (%)')
        ax_accuracy.legend()


def update_progress_bar(pbar, epoch, train_loss, test_loss, test_accuracy):
    """
    Update the progress bar with training information.

    Args:
        pbar: tqdm progress bar object.
        epoch (int): Current epoch.
        train_loss (float): Training loss.
        test_loss (float): Test loss.
        test_accuracy (float): Test accuracy.
    """
    pbar.set_description("Epoch: {}, Loss: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.2f}%".format(
        epoch, train_loss, test_loss, test_accuracy))


def compute_accuracy(output, target):
    """
    Compute accuracy given the output and target labels.

    Args:
        output (np.ndarray): Model output.
        target (np.ndarray): Target labels.

    Returns:
        float: Accuracy as a percentage.
    """
    return 100*np.mean(np.argmax(output, axis=1) == np.argmax(target, axis=1))



def shuffle_data(data, labels):
    """
    Shuffle the data and corresponding labels.

    Args:
        data (np.ndarray): Input data.
        labels (np.ndarray): Corresponding labels.

    Returns:
        np.ndarray: Shuffled data.
        np.ndarray: Shuffled labels.
    """
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    return data[indices], labels[indices]

def download_mnist(path):
    """
    Downloads the MNIST dataset files if they don't already exist in the specified path.

    Parameters:
    path (str): The directory where the dataset files will be stored.

    Returns:
    None
    """
    if not os.path.exists(path):
        os.mkdir(path)
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    for file in files:
        #check if file exists
        if os.path.exists(os.path.join(path, file)):
            print(f"File {file} already exists. Skipping download.")
            continue
        url = (base_url + file).format(**locals())
        print("Downloading " + url)
        try:
            urllib.request.urlretrieve(url, os.path.join(path, file))
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    print("Download complete.")

def load_mnist(path, kind="train"):
    """
    Loads the MNIST dataset from the specified path and returns images and labels. 
    Only images with labels in targets are returned.

    Parameters:
    path (str): The directory where the dataset files are stored.
    kind (str): "train" for training data, "test" for test data.

    Returns:
    images (numpy.ndarray): An array containing image data.
    labels (numpy.ndarray): An array containing corresponding labels.
    """
    if kind == "test":
        kind = "t10k"
    labels_path = os.path.join(path, "{}-labels-idx1-ubyte.gz".format(kind))
    images_path = os.path.join(path, "{}-images-idx3-ubyte.gz".format(kind))
    try:
        with gzip.open(labels_path, "rb") as lbpath:
            # Load labels data into a numpy array
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        with gzip.open(images_path, "rb") as imgpath:
            # Load images data into a numpy array and reshape it
            # Each image is originally 28x28 pixels, so they are flattened into a 1D array of 784 elements
            #images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)

        return images, labels
    except FileNotFoundError:
        print(f"Error: Dataset files for {kind} not found in {path}. Please run download_mnist to fetch the dataset.")

def plot_test_predictions(data, predictions, target):
    '''Plot 10 of the test data along with their predictions and target labels.

    Args:
        data (np.ndarray): Test data.
        predictions (np.ndarray): Predictions of the model.
        target (np.ndarray): Target labels.
        n_samples (int, optional): Number of samples to plot. Defaults to 10.
    '''
    # Get n_samples random indices
    indices = np.random.choice(data.shape[0], 10, replace=False)

    # Plot the images
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle("Test Predictions", fontsize=16)
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[indices[i]].reshape(28, 28), cmap="binary")
        ax.set(title=f"Label: {target[indices[i]]}, Prediction: {predictions[indices[i]]}")
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()


# zaladowanie danych cifar10 z https://github.com/snatch59/load-cifar-10
def load_batch(f_path, label_key='labels'):

    with open(f_path, 'rb') as f:
        if sys.version_info < (3,):
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

def load_cifar10(path, negatives=False):

    num_train_samples = 50000

    x_train_local = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train_local = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train_local[(i - 1) * 10000: i * 10000, :, :, :],
         y_train_local[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test_local, y_test_local = load_batch(fpath)

    y_train_local = np.reshape(y_train_local, (len(y_train_local), 1))
    y_test_local = np.reshape(y_test_local, (len(y_test_local), 1))

    if negatives:
        x_train_local = x_train_local.transpose(0, 2, 3, 1).astype(np.float32)
        x_test_local = x_test_local.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        x_train_local = np.rollaxis(x_train_local, 1, 4)
        x_test_local = np.rollaxis(x_test_local, 1, 4)

    return x_train_local, y_train_local, x_test_local, y_test_local


if __name__ == "__main__":

    #MNIST data
    """
    download_mnist("data")
    X_train, y_train = load_mnist("data", kind="train")
    X_test, y_test = load_mnist("data", kind="test")
    """

    #CIFAR data
    data_path = 'data_cifar\cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_cifar10(data_path)


    np.random.seed(42)

    learning_rate = 0.01
    epochs = 25
    batch_size = 1000


    X_train = X_train / 255
    X_test = X_test / 255
 
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]



    model = Full_Net(8192, 10,SGD(learning_rate),learning_rate,[128, 64], Softmax, Relu)
    model_trainer = ModelTrainer(model)


    trained_model = model_trainer.train_model(X_train, y_train, X_test, y_test, epochs, batch_size)


    test_predictions = trained_model.forward(X_test)
    test_predictions = np.argmax(test_predictions, axis=1)
    test_labels = np.argmax(y_test, axis=1)
    plot_test_predictions(X_test, test_predictions, test_labels)


