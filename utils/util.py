import argparse
import sys
import os
import shutil
import numpy as np
import datetime
import logging
import signal
import glob
import torchvision
from PIL import Image
import matplotlib.pyplot as plt




def get_args():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-c', '--config',
                        default=None,
                        type=str,
                        help='config file path (default: None)')

    parser.add_argument('-r', '--resume',
                        default=None,
                        type=str,
                        help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    return args

def logger_init(config, log_level):
    """Initialize logger

    Arguments:
        config:
            Configuration
        mode:
            train, test, val
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

    log_file_path_name = os.path.join(config['output_path'], 'log_' + timestamp)
    # log_file_path = os.path.join(config.['output_path'], config.exp_name, config.mode.lower())
    # os.makedirs(config['output_path'], exist_ok=True)

    # logFormatter = logging.Formatter("%(asctime)s | %(filename)20s:%(lineno)s | %(funcName)20s() | %(threadName)-12.12s | %(levelname)-5.5s | %(message)s")
    logFormatter = logging.Formatter("%(asctime)s | %(filename)20s:%(lineno)s | %(funcName)20s() | %(message)s")

    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(log_file_path_name)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(log_level)




def save_model(config):
    """Save model

    Arguments:
        config:
            Configuration
    """
    timestamp_end = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

    output_name = '{}_output'.format(timestamp_end)
    logging.debug('output_name {}'.format(output_name))

    shutil.move(config.output_path, output_name)
    logging.debug('Model Saved {}', output_name)


def signal_handler(config):
    """Signal Handler for Ctrl+C

    Arguments:
        config:
            Configuration
    """
    def sig_hdlr(*args):
        text = input("Save Model?")
        if text == 'y' or text == 'Y':
            save_model(config)
            exit(0)
        else:
            timestamp_end = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
            logging.debug('timestamp_end {}'.format(timestamp_end))
            exit(0)

    signal.signal(signal.SIGINT, sig_hdlr)



def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    NOTE: Input should be a list
    """
    if type(dirs) is not list:
        print('Input is not a list')
        exit(-1)

    try:
        for dir_ in dirs:
            os.makedirs(dir_, exist_ok=True)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def remove_dirs(dirs):
    """
    WARNING: Input should be a list otherwise it will delete /
    """
    if type(dirs) is not list:
        print('Input is not a list')
        exit(-1)

    try:
        for dir_ in dirs:
            if os.path.exists(dir_):
                print('Deleting {}'.format(dir_))
                shutil.rmtree(dir_)
        return 0
    except Exception as err:
        print("Removing directories error: {0}".format(err))
        exit(-1)


def shuffle_data(a, b):
    a = np.array(a)
    b = np.array(b)
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%} [{1}/{2}]".format(pct_complete, count, total)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_images(images):
    print('plot_images')
    # image = images[0]
    # images = np.transpose(images, (0, 2,3,1))

    image_idx = [np.random.randint(len(images)) for i in range(10)]

    for idx in image_idx:
        image_tensor = images[idx]
        image_np = image_tensor.numpy()
        # plt.imshow(image_np)


        plt.figure()
        # imshow needs a numpy array with the channel dimension as the the last dimension
        plt.imshow(image_tensor.numpy().transpose(1, 2, 0))
        plt.show()

        # # pil_to_tensor = transforms.ToTensor()
        # tensor_to_pil = torchvision.transforms.ToPILImage()
        # image_pil = tensor_to_pil(image_tensor)
        # image_pil.save("your_file.jpeg")

        # image_pil = Image.fromarray(image_np)
        # image_pil.save("your_file.jpeg")



