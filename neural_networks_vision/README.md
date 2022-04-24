## Convolutional Neural Network for Vision Tasks

In recent years, the convolutional neural network (conv net) has been the most popular neural network for computer vision tasks like classification, detection and segmentation. The conv net is more suitable for images as compared to a multi-layer perceptron as it uses kernels (3 x 3 being the most popular) to perfrom a convolution on the image. The kernel has a receptive field that allows it to have an understanding of the surrounding pixels.  

Goals for this segment of my learning would be to gain familiarity with working both in Tensorflow and PyTorch as these are the 2 largest machine learning frameworks we have today. I aim to understand VGG and MobileNet V2. 

1. VGG comprises of pure convolutional layers with a fully connected layers at the top to enable classification. It has a very simple and easy to understand structure and produced state of the art results for its time.  

2. MobileNet V2 stood out to me because it managed to achieve similar top 1 performance on the ImageNet dataset (71.3%) but with a low latency (25.9ms vs 69.5ms inference time on CPU) and a much reduced number of parameters as compared to VGG16 (3.5M vs 138.4M). The model architecture was also designed with efficiency in mind, to be able to run on low powered mobile devices.  

## Key Thoughts

1. We should use transfer learning for vision tasks as both Tensorflow and PyTorch offer CNNs that have been trained on large datasets and the kernels learned can generalise to our dataset. We start fine tuning on our dataset by freezing all the convolutional layers and if the generalisation is not good, we have the option to unfreeze more of the network to allow the network leeway to fit to our data.

2. The reduce on plateau learning rate scheduler can help in the training process by reducing the learning rate automatically when a particular metric (e.g. validation loss) reaches a plateau.

3. We utilise data augmentation, e.g. random horizontal flips, random affine transformation (rotation, translation and scaling) to provide some randomness in our training process to ensure that our model does not overfit to our dataset. Always make sure that the data augmentation makes sense for your dataset, e.g. random flips (horizontal / vertical) do not make sense for handwritten digits as a flipped digit is not something we want the model to learn. 

4. We use batch normalisation (for the convolution layers) and dropouts (for the fully connected layers) during training to prevent overfitting.

5. There is no PyTorch in-built method for stratification when we want to split our image dataset into train, test and validation. This can be achieved by passing the class index to Sci-Kit Learn's train-test split function to achieve stratification.  

6. If a GPU is available, it is typically faster for training than on the CPU as a GPU is suited to perform computations in parallel. When training on a GPU, there is additional overhead to move the files from the CPU to the GPU, hence we try to maximise the GPU ram usage by increasing the batch size as much as possible before we get an error that says we are out of ram. We can also monitor the GPU ram usage in the Windows Task Manager.

7. We aggregate the training and validation loss over all the batches. We divide the training and validation loss by the number of samples in each set to get the average loss per sample for comparison. 

8. MobileNet implements the following ideas:

    a) Depth-wise separable convolution to reduce the computations required. The 3x3 convolution is split into: 1) a 3x3 kernel which looks at a single channel and 2) a 1x1 kernel which looks at all the channels 

    b) 

## References

1. [DeepLearningAI Conolutional Neural Network Course](https://www.youtube.com/watch?v=ArPaAX_PhIs&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)

2. [VGG-16: A simple implementation using Pytorch](https://medium.com/@tioluwaniaremu/vgg-16-a-simple-implementation-using-pytorch-7850be4d14a1). This is a good reference for those who want to know how to implement VGG-16 from scratch.

3. [ImageNet Class Labels](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a). I utilised the class labels for ImageNet from this Github repository.

4. [Pytorch Models and Pre-Trained Weights](https://pytorch.org/vision/stable/models.html). Pytorch's official documentation on the pre-trained weights for vision models. 

5. [Tensorflow Models and Pre-Trained Weights](https://keras.io/api/applications/)

## Datasets

1. [Classification: Animals 151](https://www.kaggle.com/sharansmenon/animals141)

2. [Classification: Brain Tumour Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection). I noticed that there was quite a lot of cleaning to be done for this dataset.

    1) The cropping was inconsistent and quite wide on some of the samples, all samples were cropped close to the brain

    2) Overlays (text, rulers, arrows) that were not on the image of the brain itself was removed.

    3) Images where there were overlays on the brain itself were discarded

