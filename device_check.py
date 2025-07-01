from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# 或更简洁的版本：
import tensorflow as tf
print("GPU可用性:", tf.test.is_gpu_available())
print("检测到的设备:", tf.config.list_physical_devices('GPU'))
