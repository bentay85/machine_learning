## Introduction

This repository records my exploration of deployment using TensorRT. I was doing everything here on a Windows machine with a GTX 1660Ti graphics card. Hence, I was using the official Tensorflow docker container from Nvidia. 

A similar approach could be taken on an edge device (not tested yet), e.g. Jetson Nano but directly on the board without the use of the docker container as the required packages would already be included with the Jetpack installation. 

## Docker Container

Download Docker for Windows and install it for your machine from here: https://www.docker.com/

Ensure that docker is running before running the following command in a cmd prompt: 
```
docker pull nvcr.io/nvidia/tensorflow:21.05-tf2-py3
```
Version 21.05 of the container selected specifically for Tensorflow 2.4 and TensorRT7. There were errors when I tried using the latest docker containers.

Run the docker container with the following command:
```
docker run --gpus all -it --rm --mount type=bind,src="C:\PATH_TO_FOLDER",dst="/workspace/tensorflow/" nvcr.io/nvidia/tensorflow:21.05-tf2-py3
```
1) --gpu all enable GPU access in the docker container. 
2) -it runs in interactive mode. 
3) --rm removes the container after exit. 
4) --mount allows you to map a folder on your desktop to a location within the container. Replace "C:\PATH_TO_FOLDER" to the path to which you cloned this repository.

Within the docker container, we need to install some dependencies:
```
pip install pycuda onnxruntime tf2onnx scikit-learn
```

## Commands to Generate Models
Within the docker container, run the following command to train the tensorflow model on the Fashion MNIST dataset. You can replace this process with training your own tensorflow model.
```
python mnist.py
```

There are a few types of models that we are generating to check the relative performance.

1) Tensorflow-TensorRT. This is the most direct way of converting from Tensorflow to TensorRT. Lines 22-23 in tftrt_modelconvert.py (included in this repository) define the quatisation precision. Please check that the precision you are selecting is supported by your device. 
```
python tftrt_modelconvert.py
```

2) ONNX. ONNX is an open format built to represent machine learning models. 
```
python -m tf2onnx.convert --saved-model tf_model/ --output onnx_model/model.onnx
```
3) ONNX to TensorRT. This is another way of generating a TensorRT model. --shapes argument required to run the full batch of 10000 test images simultaneously
```
mkdir trt_model
trtexec --onnx=onnx_model/model.onnx --shapes=flatten_input:10000x28x28 --saveEngine=trt_model/model_engine.trt
```

## Testing the models

To test the models, run the following command.
```
python check_times.py
```

## References

Some references I referred to during the process:

1) https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_21-12.html#rel_21-12

2) https://www.tensorflow.org/install/gpu

3) https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow