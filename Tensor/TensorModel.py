from Tensor.tensor_imports import *
import Tensor.DataProcessor as DataProcessor

class TensorModel:

    model_data: DataProcessor = DataProcessor()

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
        # Get full image dataset
        
