import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score

import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

import timeit

#load and normalise train and test sets
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
 
train_images = train_images / 255.0
test_images = test_images / 255.0

test_images_onnx=test_images.astype(np.float32)

model = tf.keras.models.load_model('tf_model/')

model.summary()

print("CPU")

with tf.device("/cpu:0"):
    starttime = timeit.default_timer()
    results = model.evaluate(test_images, test_labels, batch_size=16384)
    print("Time Taken: ", timeit.default_timer() - starttime)
    print("Test Accuracy: ", results[1])  #results is a tuple [loss, accuracy]
    print(" ")

print("Regular GPU (Ignore First Run)")
#First evaluation is slow
results = model.evaluate(test_images, test_labels, batch_size=16384)

starttime = timeit.default_timer()
results = model.evaluate(test_images, test_labels, batch_size=16384)
print("Time Taken: ", timeit.default_timer() - starttime)
print("Test Accuracy: ", results[1])  #results is a tuple [loss, accuracy]
print(" ")

from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import convert_to_constants

print("TF-TRT FP32")
saved_model_loaded = tf.saved_model.load("tftrt_model/fp32/", tags=[tag_constants.SERVING])
graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
frozen_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)

test_images = tf.convert_to_tensor(test_images, dtype = float)
output = frozen_func(test_images)[0].numpy()

starttime = timeit.default_timer()
output = frozen_func(test_images)[0].numpy()
print("Time Taken: ", timeit.default_timer() - starttime)

output = np.argmax(output, axis=1)
acc_score = accuracy_score(test_labels, output)

print("Accuracy Score: " + str(acc_score))
print(" ")

print("TF-TRT FP16")
saved_model_loaded = tf.saved_model.load("tftrt_model/fp16/", tags=[tag_constants.SERVING])
graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
frozen_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)

test_images = tf.convert_to_tensor(test_images, dtype = float)
output = frozen_func(test_images)[0].numpy()

starttime = timeit.default_timer()
output = frozen_func(test_images)[0].numpy()
print("Time Taken: ", timeit.default_timer() - starttime)

output = np.argmax(output, axis=1)
acc_score = accuracy_score(test_labels, output)

print("Accuracy Score: " + str(acc_score))
print(" ")

print("ONNX")

session = onnxruntime.InferenceSession("onnx_model/model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
#print(input_name)
#print(output_name)

starttime = timeit.default_timer() 
result = session.run([output_name], {input_name: test_images_onnx})
print("Time Taken: ", timeit.default_timer() - starttime)
#print(result[0].shape)

output = np.argmax(result[0], axis=1)
acc_score = accuracy_score(test_labels, output)

print("Accuracy Score: " + str(acc_score))
print(" ")

print("ONNX-TRT")

TRT_LOGGER = trt.Logger()

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    #print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
    
def infer(engine, input_image):

    with engine.create_execution_context() as context:
        # Set input shape based on image dimensions for inference
        context.set_binding_shape(engine.get_binding_index("flatten_input"), (10000, 28, 28))
        # Allocate host and device buffers
        bindings = []
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(input_image)
                input_memory = cuda.mem_alloc(input_image.nbytes)
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))

        stream = cuda.Stream()
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        # Synchronize the stream
        stream.synchronize()
    
    return output_buffer

engine_file = "trt_model/model_engine.trt"

with load_engine(engine_file) as engine:
    #print(engine.get_binding_name(0))
    #print(engine.get_binding_name(1))
    starttime = timeit.default_timer()
    output_buffer = infer(engine, test_images_onnx)
    print("Time Taken: ", timeit.default_timer() - starttime)

with load_engine(engine_file) as engine:
    #print(engine.get_binding_name(0))
    #print(engine.get_binding_name(1))
    starttime = timeit.default_timer()
    output_buffer = infer(engine, test_images_onnx)
    print("Time Taken: ", timeit.default_timer() - starttime)
    
output = output_buffer.reshape((10000,10))
output = np.argmax(output, axis=1)
acc_score = accuracy_score(test_labels, output)
print("Accuracy Score: " + str(acc_score))