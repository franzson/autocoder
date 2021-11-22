# autocoder

The autocoder package is an implementation of a variational autoencoder––a neural network capable of learning a spectral representation of a soundfile and synthesizing a novel output based on the trained model. It provides an easily extendable ecosystem to assist with the experimentation and development of sound software and hardware based on the underlying neural network architecture.

# installation

For basic usage, you will need python 3.7+, Tensorflow and Numpy
Tensorflow 2.4+
	Mac OS 12+: https://developer.apple.com/metal/tensorflow-plugin/
	Other: https://www.tensorflow.org/install
For a lighter version, the generative parts of the code can be used with a tflite_runtime install
	https://www.tensorflow.org/lite/guide/python
Numpy: python3 -m pip install numpy

Depending on features required, you will need some or all of the following:
Scipy: python3 -m pip install scipy
PyAudio: python3 -m pip install pyaudio
Python Speech Features: python3 -m pip install python-speech-features
librosa: python3 -m pip install librosa

On Mac OS 12+, some packages might be easier to install using conda.

Download the code and run one of the following:
python3 ./autocoder_analyze.py -h
	This file contains analysis and training functions.
python3 ./autocoder_generate.py -h
	This file contains various generative functions.
python3 ./autocoder_remote.py -h
	This file contains and OSC interface to the model.
python3 ./autocoderlib.py -api
	This file contains the actual model and helper functions.

A very simple Max/MSP demo patch is included in /maxmsp. This patch requires spat5 (https://forum.ircam.fr/projects/detail/spat/) to be installed.

A google colab (https://colab.research.google.com/) training script is included in /colab. You can use a chrome extension called 'Open in Colab'  to move the script over.  It is highly recommended to use colab for any heavier training as cpu training can be relatively slow.

# autoencoders

An autoencoder is a neural network that takes an input, runs it through one or more hidden layers and reproduces the input as accurately as it can. It can be described as a same-in-same-out structure where a compressed representation of the training data is learnt by the model.

The network consists of two separate parts: an encoder that takes the training data and produces a latent vector encoding the training data in a lower number of dimensions; and a decoder that  reconstructs the original input based on the latent vector. The two parts are trained at the same time, using the error between the input and output as a metric to adjust the weights of the hidden layers and in the process, changing how the input maps onto the latent vector.

![alt text](https://github.com/franzson/autocoder/blob/main/images/autocoder.001.jpeg)

The encoder and decoder can be used separately from one another to either encode a sound into a compressed format––e.g. for sound similarity judgement––or to decode an arbitrary or synthetic latent vector, generating new spectra based on the spectral space of the training data.

# variational autoencoder

The latent vector space of an autoencoder can easily become lumpy as the model is blind to the internal structure of the latent vector, and as a result, minor changes in the latent space can produce wild variations in the output of the decoder, rendering it hard to control as a generative tool and making distances between any two latent vectors dependent on the magnitude  of their values (i.e. the distance between 0 and .1 might be 10x larger  than the distance between 0.9 and 1).

By constraining the model to fit a probability distribution in the latent vector layer, the latent vector can be forced to describe any input as a point on a smooth/continuous plane. The model is in a sense representing a smooth surface in the input dimensions onto which each point in the training data can be mapped via its latent representation.

Conversely, any point on this surface can be decoded into an output that is representative of the training data, regardless of wether it is present in the training data or not. This means we can feed arbitrary latent vectors into the model and have the decoder part of the model decode them into the space of the training data with distances within the latent space mapping relatively evenly onto distances in the output space.

For an overview of Variational Autoencoders, see Diederik P. Kingma and Max Welling (2019), “An Introduction to Variational Autoencoders”, Foundations and Trends in Machine Learning: Vol. xx, No. xx, pp 1–18. DOI: 10.1561 (https://arxiv.org/pdf/1906.02691.pdf)

# implementation

A spectral representation is extracted from an input sound by running it through an STFT analysis, producing a number of spectral frames representing the sound moment by moment. Any sequential or temporal structure that is not encoded within a frame is therefor lost.

Without further processing, spectral data is fairly useless as training data for a simple AI model, as half of the data represents the top octave of the sound which at a sample rate of 44.1kHz would be the frequencies between 11.025kHZ and 22.05kHz, and half of the rest of the data the octave below that, so three quarters of each frame in the training data would encode the frequencies from c.a. 5500Hz and up which is a relatively small part of how we hear the sound. Without the ability to tell which frequencies are important to our hearing, the neural network will then spend most of its time and energy on modeling the finer details of noise present in the training data rather than the frequencies that are meaningful to how we hear the sound.

![alt text](https://github.com/franzson/autocoder/blob/main/images/autocoder.002.jpeg)

A similar problem arises with phase. The spectral amplitude, or the relationship between amplitudes along the spectrum are absolute values, while the phase can be rotated and cuts off  as it crosses either Pi or -Pi, making both raw phase and phase–difference hard to train on without some clever cooking of the raw data. [This shouldn't be hard to add into the model and I would be very interested in any thoughts for solutions on this problem].

To work around these two problems, the phase information is thrown out, and the amplitude of the spectrum is converted to a mel-frequency cepstrum (MFCC), a linear transform of the log distribution of energy in the spectrum. In the MFCC, each octave is represented by an equal number of values so that the noise modeling problem becomes moot.

Converting the data to MFCC also allows for a large compression of the input spectrum, from 8192 points for an FFT window size of 16384 down to 512 points. Another advantage of the MFCC conversion is that the mel space can be unpacked to any arbitrary FFT window size, allowing for the input and output FFT size to be different from one another with limited loss of information.

The data set is then normalized and fed into the encoder network. 

The software implements two network architectures. The architectures are somewhat arbitrary and designed through trial and error and could be improved upon. The first is a 'shallow' single hidden layer model that learns general features of the model efficiently and fast. The second is a 'deep' model, which in theory should encode more details, but at the cost of re–introducing slight lumpiness, as the variability in how features get encoded in the different layers seems to sometimes compound.

![alt text](https://github.com/franzson/autocoder/blob/main/images/autocoder.003.jpeg)

The (first) hidden layer explodes the input data into a layer with a larger number of dimensions, in this case a vector of 1000 points, allowing for better feature separation (CITATION). A hidden layer size above roughly 1.25x  and lower than 2x the size of the input layer should be safe. 

The latent vector layer of the encoder has eight values, each encoding some aspect of the input data. By feeding the training data through the encoder after training,  min and max factors can be extracted to scale the range of values in the latent vector that correspond to the min and max values in the encoded training data to the range from 0 to 1.

The eight values are dependent on each other, changing one affects what the other 7 values represent, e.g. a value of .1 on in one layer doesn't always map onto the same feature in the output of the decoder.

# applications

By feeding arbitrary values as input into the latent vector layer, the encoder returns a new mel frame that represents an unseen point within the spectral space of the training data. The frame is translated from mel to  spectrum via a matrix multiplication––this is the main bottleneck in the algorithm––and can then be used in any number of ways, e.g. for synthesis or convolution, as an impulse response in a hybrid reverb, or for cross–synthesis.

#1. generative synthesis

A continuous soundscape––based upon the training data––can be produced by generating a random walk for each value in the latent vector and then decoding the latent vector into a spectral frame. The resulting spectrum is then synthesized by randomizing the phase and taking the inverse FFT of the frame.

If the step size of the random walk produces undesirable discontinuities in the output, spectral domain low pass filtering can be applied. 

	EXAMPLES

# 2. REVERB/FILTERING

The network can also be used to convert a discrete input space into a continuous one. For example, a set of reverb impulses (the amplitude, not phase) can be fed as training data to the model, producing a continuous space that can be used for the filtering stage in a hybrid reverb algorithm. 

This generates new unheard spaces in-between the original impulses, as well as offering the ability to dynamically morph between discrete spaces.

	EXAMPLE

Similarly, a generative virtual resonance body for an instrument could be produced from a number of real-world and synthetic responses, creating new hybrids as well as hybrid analog/virtual instruments where the resonant response of the instrument shifts dynamically in time.

Models used for generative synthesis can also be used for the filtering stage, resulting in a cross synthesis where the carrier signal is generative and dynamic within the space defined by the training data.

	EXAMPLE

By whitening the modulator spectrum––i.e. normalizing the frame to 0 to 1 and then setting it to a fractional power––the need for loud bins to align for a signal to be produced can be mitigated and a more forgiving imprint of the modulator on the carrier can be produced. 

	EXAMPLE

By stringing a few of these together, extended reverb tails that change and morph in–time can be achieved.

# 3.autocoding

A more direct form of cross synthesis can be produced by feeding a different input sound into the encoder, producing a latent representation of that new input sound as heard by the AI. Since the AI only knows how to hear things based on its training data, a hybrid sound can in theory be produced. In reality, it is highly unlikely that the latent vector representations even as much as overlap, so the latent representation of the new input must be offset, clipped and scaled. This representation can then be decoded, effectively skinning the input sound with parts of the sound that the original model is based on.

	EXAMPLE

Since this isn't a theoretically rigorous approach in the first place, the mapping of the different dimensions of the latent vector of the input sound can be scrambled and the dimensions themselves inverted, producing numerous variants of the original input.

	EXAMPLES

# 4. further work

Given that the latent vector encodes a smooth representation of the training data, distances in the latent layer can be used as proxies for distances in the training data, allowing us to calculate the similarities between any two sounds––or rather between the spectral representations of any two sounds––by calculating the euclidian distance between their latent representations. These similarity judgements could be used for concatenative synthesis, granular synthesis based on similarities between spectral frames, as well as larger scale compositional judgements based on relationships between sounds along the dimensions of the latent vector.

Training parameters can also be used creatively. In my experience, smaller batch sizes––i.e. the number of examples from the training set shown to the model at a time––tend to produce models that focus on individual sound components, while larger batch sizes are less detail oriented and try to find shortcuts to represent the data. Taking advantage of the larger model shortcutting, latent spectral/harmonic structures between the sounds can emerge––i.e. feeding a complete chromatic set of all string instruments  as a training set and training with a large batch size seems to lead to a harmony based on the overtones present in the string sounds to emerge rather than the ability to produce each pitch separately, although this is not the case for every round of training.

This code for granular synthesis and distance calculations is still under very active development and is therefor still buggy, and should not be expected to stay consistent or even return consistent returns.

The similarity algorithm can also be repurposed to operate on time-series rather than spectral data which then can be used to generate control envelopes using concatenative or granular synthesis to resynthesize the output of the model into a continuous signal.

Currently we are developing a synthetic body for the Halldorophone as well as a eurorack module based implementation of the convolution algorithm.

As the network architecture is fairly simple, training times on google colab are roughly equal to the duration of the input sound for a lower resolution model with batch size set to 256, and around 3x the duration of the input for a more detailed model. With batch size set to 4096, the training time of a lower quality model drops to less than 25% of the original duration.

# usage

The various scripts use the name of the original sound file as a way of keeping track of various specialized files between functions.

To create a training set:
python3 ./autocoder_analyze.py -a 'input_file.wav'

To create a training set from a folder of files with n frames from each file, feed it the folder and a number:
python3 ./autocoder_analyze.py -a 'folder' n
	(Creates a dummy file named combined.wav in the input folder for use with training).

To create a training set where each piece of data is the average spectrum of each file in a folder:
python3 ./autocoder_analyze.py -a 'folder' average
	(Creates a dummy file named combined.wav as well as a file called combined.txt containing the filenames 
	 in the folder).

To create a training set from the first FFT frame of each file in a folder:
python3 ./autocoder_analyze.py -a 'folder' first
	(Creates a dummy file named combined.wav).

To train (training on Google Colab is very strongly recommended):
python3 ./autocoder_analyze.py -t 'input_file.wav' deep[0/1]
	(Regression patience, delta and learning rate can also be set on the command line, see -h for more information)

To calculate distances across the frames of the training set.
python3 ./autocoder_analyze.py -e 'input_file.wav'
Followed by:
python3 ./autocoder_analyze.py -d 'input_file.wav' n_of_most_similar
	(n_of_most_similar can be any number between 1 and the number of files in the folder - 1).

The resulting models can then be used with either ./autocoder_generate.py and ./autocoder_remote.py

To run the max/msp examples, first run python3 ./autocoder_remote.py 4013 4061 and then load the patch.

# training

The training parameters are set to construct a relatively good generative model from a well structured input, i.e. a song or another piece where pitch relationships are concurrent. For unstructured models, i.e. large sample sets of shorter sounds–– try using deep learning and increase the amount of training by increasing the regression patience.

Lowering the batch size will allow the model to learn more detail at the cost of larger relationships, i.e. learns sine wave like features rather than sound mass like features. Higher batch sizes train much faster and are good for quickly getting a sense of what the model will learn.
