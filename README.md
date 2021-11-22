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

![alt text](https://github.com/franzson/autocoder/blob/main/images/autocoder.001.jpeg)

![alt text](https://github.com/franzson/autocoder/blob/main/images/autocoder.002.jpeg)

![alt text](https://github.com/franzson/autocoder/blob/main/images/autocoder.003.jpeg)
