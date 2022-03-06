## Audio Keyword Recognition

This is a very small project for me to understand how to work with audio data and train a classifier to recognise the keywords spoken. The basic idea that I will be implementing is converting the waveform of the sound clip into the frequency domain using a Fourier Transform. Then I will feed the Fourier transform into a neural network for classification. I am intending to start with 3 classes: ambient sound, forward command spoken and backward command spoken. This is useful for robotics applications where you want to give commands to the robot to move forewards or back.  

## References

1. [Assembly AI: Getting Started With Torchaudio. ](https://www.youtube.com/watch?v=3mju52xBFK8)

2. [3Blue1Brown: But what is the Fourier Transform? A visual introduction. ](https://www.youtube.com/watch?v=spUNpyF58BY) This was an excellent video to gain an fundamental understanding of what a fourier transform is. 

3. [SciPy Fast Fourier Transform Documentation](https://docs.scipy.org/doc/scipy/tutorial/fft.html)

## Libraries Used

1. [Pytorch, Torch Audio](https://pytorch.org/audio/stable/index.html)

2. [IPython](https://ipython.org/): For Playback of audio

2. [FFMPEG](https://ffmpeg.org/): Used to transcode MP3 to Wav

3. [Scipy](https://scipy.org/): For Fast Fourier Transform

Please see conda.yml for the dependencies used. I am on a Windows system so these should be everything you need to run this project. Do let me know if I missed out any requirements.

## Dataset

The dataset used in this training was synthetically generated using [TTSMP3.com](https://ttsmp3.com/), converted to wav using FFMPEG for easy loading into Torch Audio.

Audio used for ambient sounds was taken from [SoundJay.com](https://www.soundjay.com/ambient-sounds.html). Unfortunately, due to the license on the SoundJay website, I am unable to share the audio files that I have created. However, you can use the jupyter notebook to generated your own training samples using ambient sounds you download from SoundJay.com or record yourself.

Numpy was used to randomly truncate the command given so as to simulate a fixed 1s window of audio recording where your command could be truncated in real life. Random ambient sounds were also added to make the samples more realistic. 