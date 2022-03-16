Material for Learning

1. [DeepLearningAI Conolutional Neural Network Course](https://www.youtube.com/watch?v=ArPaAX_PhIs&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)

2. [Pytorch Models and Pre-Trained Weights](https://pytorch.org/vision/stable/models.html). Pytorch's official documentation on the pre-trained weights for vision models.  

3. [VGG-16: A simple implementation using Pytorch](https://medium.com/@tioluwaniaremu/vgg-16-a-simple-implementation-using-pytorch-7850be4d14a1). I utilised the VGG16 model from this website. 

4. [ImageNet Class Labels](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a). I utilised the class labels for ImageNet from this Github repository.

Datasets

1. [Classification: Animals 151](https://www.kaggle.com/sharansmenon/animals141)

2. [Classification: Brain Tumour Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection). I noticed that there was quite a lot of cleaning to be done for this dataset.

    1) The cropping was inconsistent and quite wide on some of the samples, all samples were cropped close to the brain

    2) Overlays (text, rulers, arrows) that were not on the image of the brain itself was removed.

    3) Images where there were overlays on the brain itself were discarded

