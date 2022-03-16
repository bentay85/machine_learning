## Audio Keyword Recognition

This is a very small project for me to understand how to work with audio data and train a classifier to recognise the keywords spoken. The basic idea that I will be implementing is converting the waveform of the sound clip into the frequency domain using a Fourier Transform. Then I will feed the Fourier transform into a neural network (multi-layer perceptron) for classification. I am intending to train my model to recognise with 3 classes: ambient sound, forward command spoken and backward command spoken. This is useful for robotics applications where you want to give commands to the robot to move forwards or back and can be easily extended to more commands by changing the output layer of the multi-layer perceptron.

## Key Thoughts

1. Synthetic to real translation. I use synthetically generated voice from a text to speech engine to train the neural network but tested the resulting model on my own voice and the model was able to generalise.

2. Generating traning samples from limited data. I create multiple training samples by mixing and matching the voice command with background sounds and also offsetting the crop on the commands such that up to half the voice command can be cut off either from the front or back. This is to simulate the possibility of half a command being captured in a real voice streaming scenario where we take 1s snippets for inference. 

3. Keeping only the fourier transform vector. I standardised the sampling frequency and the sample length so that the frequency vector of the fourier transform is the same across all samples and can be ignored. We use only the fourier transform vector for training and inference. 

4. Deployment considerations. As this will be a low power application, probably used within an embedded processor (Raspberry Pi or Arduino Nano BLE Sense) on a robot, the model has to be small and lightweight, hence a simple multi-layer perceptron was chosen. 3 fully connected layers define the network and the hidden layers have 32 neurons each. 

## References

1. [Assembly AI: Getting Started With Torchaudio. ](https://www.youtube.com/watch?v=3mju52xBFK8)

2. [3Blue1Brown: But what is the Fourier Transform? A visual introduction. ](https://www.youtube.com/watch?v=spUNpyF58BY) This was an excellent video to gain an fundamental understanding of what a fourier transform is. 

3. [SciPy Fast Fourier Transform Documentation](https://docs.scipy.org/doc/scipy/tutorial/fft.html)

## Libraries Used

1. [Pytorch, Torch Audio](https://pytorch.org/audio/stable/index.html)

2. [IPython](https://ipython.org/): For Playback of audio

3. [FFMPEG](https://ffmpeg.org/): Used to transcode MP3 to Wav

4. [Scipy](https://scipy.org/): For Fast Fourier Transform

5. [PyAudio](https://pypi.org/project/PyAudio/): Used to record sound from the mic

Please see conda.yml for the dependencies used. I am on a Windows system so these should be everything you need to run this project. Do let me know if I missed out any requirements.

## Dataset

The dataset used in this training was synthetically generated using [TTSMP3.com](https://ttsmp3.com/), converted to wav using FFMPEG for easy loading into Torch Audio.

Audio used for ambient sounds was taken from [SoundJay.com](https://www.soundjay.com/ambient-sounds.html). Unfortunately, due to the license on the SoundJay website, I am unable to share the audio files that I have created. However, you can use the jupyter notebook to generated your own training samples using ambient sounds you download from SoundJay.com or record them yourself.