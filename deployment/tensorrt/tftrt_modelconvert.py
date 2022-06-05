from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
from tensorflow import keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

FP32_SAVED_MODEL_DIR = "tftrt_model/fp32"

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=trt.TrtPrecisionMode.FP32)

converter = trt.TrtGraphConverterV2(input_saved_model_dir="tf_model/",conversion_params=conversion_params)
converter.convert()

converter.save(FP32_SAVED_MODEL_DIR)


FP16_SAVED_MODEL_DIR = "tftrt_model/fp16"

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=trt.TrtPrecisionMode.FP16)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir="tf_model/",
conversion_params=conversion_params)
converter.convert()

converter.save(FP16_SAVED_MODEL_DIR)
