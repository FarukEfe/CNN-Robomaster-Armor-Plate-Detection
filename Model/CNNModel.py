from imports import *

class CNNModel:

    def __init__(self) -> None:
        # Set logger for debug
        tf.get_logger().setLevel("INFO")
        tf.autograph.set_verbosity(0)
        tf.get_logger().setLevel(logging.ERROR)
        # Set tensor environment
        SEED = 12345
        np.random.seed(12345)
        tf.random.set_seed(12345)
        os.environ["PYTHONHASHSEED"] = str(SEED)
        