#sudo /usr/sbin/nvpmodel -m
#jtop
# ADD A 10x AMP
# AND ADD A 5V SHIFT

# SWAP BETWEEN MODELS IN A DIRECTORY

# GET IT TO WORK WITH SD
# ADD MODE SELECTION
# ADD BUTTONS

# MAKE RENDERER AND PLAYBACK CAPABLE OF OTHER WINDOWSIZES, CURRENTLY HARDWIRED TO 16384

#############################
#          IMPORTS          #
#############################
import time
import sys
import numpy as np
if(sys.argv[1] != "-a"):
    import tensorflow as tf
    from tensorflow.python.keras import backend as K
    sess = tf.compat.v1.Session()
from scipy.signal import hann
import scipy.io.wavfile
import scipy
import math
try:
    import librosa
except ImportError as e:
    pass
from python_speech_features.base import get_filterbanks
from typing import List, Any
import platform
if(sys.argv[1] == "-p"):
    try:
        import pyaudio
    except ImportError as e:
        pass
    try:
        import pyserial
    except ImportError as e:
        pass
if((sys.argv[1] == "-o") or (sys.argv[1] == "-m")):
    try:
        import pythonosc
        from pythonosc import dispatcher
        from pythonosc import osc_server
        #from pythonosc import osc_message_builder
        from pythonosc import udp_client
    except ImportError as e:
        pass
try:
    from numba import jit
except ImportError as e:
    pass

import os


#############################
#    EDITABLE PARAMETERS    #
#############################

fftsize = 16384
windowskip = 1024    # TRAINING, NOT GENERATOR, GENERATOR IS HARD-CODED TO 4096
melsize = 512
learning_rate = .00001 #.00001
batch_size = 4096    ### SET TO 4 FOR M1, 32 FOR x86 AND 4096 FOR JETSON OR GPU
min_delta = .00001
regression_patience = 100
oscinport = 4013
oscoutport = 4061
brightness = 0.1

#############################
#     CONTROL VARIABLES     #
#############################

feedback = .25 # 1.0 = no feedback
random_max = 10. # step = .5 / random_max_step -- clip this to 1 / 10 // NEEDS TO GO EVEN LOWER
random_pow = .5 # levy flight parameter -- clip this to .1 /10. with an exponent
random_max_adjusted = random_max
random_pow_adjusted = (pow(random_pow, 3) * 9.9) + .1

#############################
#     DYNAMIC PARAMETERS    #
#############################

n_epochs = 50000           # MAXIMUM NUMBER OF EPOCHS WHILE ATTEMPTING TO CONVERGE
intermediate_dim = 1000    # HOW WIDE THE FIRST LAYER IS
encoding_dim = 8           # HOW WIDE THE MIDDLE LAYER IS

##################################
#  GLOBAL VARIABLE PLACEHOLDERS  #
#          DO NOT EDIT           #
##################################

mel_filter = 0
mel_inversion_filter = 0
input = 0
scale_amax = 1
scale_mult = 0
scale_subtract = 0
minin = 0.
maxin = 1.
vae = 0
encoder = 0
decoder = 0
deep = 0
analysismod = 1
output_frame = 0
internal_vector = np.zeros((1, 8))
brightness_ = 0

np.set_printoptions(threshold=sys.maxsize)

################################
#   GUESS OPTIMAL BATCH SIZE   #
################################

def set_batch_size():
    global batch_size

    psystem = platform.system()
    pmachine = platform.machine()
    pprocessor = platform.processor()

    if((psystem=='Darwin') & (pmachine == 'arm64')):
        print("... setting batch size to 4 for m1 mac ...")
        batch_size = 4

    elif((psystem=='Linux') & (pmachine == 'aarch64')):
        print("... setting batch size to 4096 for nvidia jetson ...")
        batch_size = 4096

    else:
        print("... setting batch size to 32 for a generic system ...")
        batch_size = 32

#############################
#           UI CODE         #
#############################

def update_feedback():  # FEEDBACK IS THE SMOOTHING FACTOR
    global feedback_adjusted
    feedback_adjusted = feedback # PASSTHROUGH FOR NOW

def update_random_max():
    global random_max_adjusted
    random_max_adjusted = random_max # PASSTHROUGH FOR NOW, SHOULD CONVERT 0-1 TO 10 - 1

def update_random_pow():
    global random_pow_adjusted
    random_pow_adjusted = (pow(random_pow, 3) * 9.9) + .1  # MAPS 0 - 1 TO .1 - 10. WITH .5 = 1.25

#############################
#          MEL CODE         #
#############################

def create_mel_filter(fft_size, n_freq_components = 64, start_freq = 300, end_freq = 8000, samplerate = 44100):
    filterbanks = get_filterbanks(nfilt=n_freq_components,
                                           nfft=fft_size, samplerate=samplerate,
                                           lowfreq=start_freq, highfreq=end_freq)
    mel_inversion_filter = (filterbanks.T[0:(int(fft_size/2))]).T
    mel_filter = np.divide(mel_inversion_filter.T, mel_inversion_filter.sum(axis=1))

    return mel_filter, mel_inversion_filter

def spectrogram_to_mel(spectrogram, filter):
    mel_spec = np.transpose(filter).dot(np.transpose(spectrogram))
    return mel_spec

def mel_to_spectrogram(mel_spec, filter, spec_thresh):
    mel_spec = (mel_spec+spec_thresh)
    uncompressed_spec = np.transpose(np.transpose(mel_spec).dot(filter))
    uncompressed_spec = uncompressed_spec / spec_thresh - 1.
    return uncompressed_spec

#############################
#       ANALYSIS CODE       #
#############################

def convertToBin(data):
    return(np.sqrt(np.add(np.multiply(data.real, data.real), np.multiply(data.imag, data.imag))))

def readwave(filename):
    wavefile = librosa.load(filename, sr = 44100, mono = True)
    print("  ####################################")
    print("  #   number of samples:  ", len(wavefile[0]))
    print("  ####################################")
    print("")
    return(wavefile[0])

def prepanalysis(size):
    global window
    global mel_filter
    global mel_inversion_filter

    window = np.zeros((1, size))
    window[0,] = hann(size)
    mel_filter, mel_inversion_filter = create_mel_filter(size, melsize, 0, 22050, 44100)

def analyze(data):

    data = np.multiply(data, window)
    fftdata = scipy.fft.rfft(data)
    ampslize = convertToBin(fftdata)
    #phase = np.angle(fftdata)
    melslize = spectrogram_to_mel(ampslize[0,0:8192], mel_filter)
    return(melslize)

def analyze_data(data, filename):
    global minin
    global maxin

    n_slizes = round(len(data)/windowskip)
    output = np.zeros((int((n_slizes - 16) / analysismod)+1, melsize))

    in_slize = np.zeros((1, fftsize))
    for i in range(0, (n_slizes - 16)):
        if(i % analysismod == 0):
            in_slize[0] = data[i * windowskip:((i*windowskip) + fftsize)]
            output[int(i / analysismod),:] = analyze(in_slize, )

    if(scale_amax == 1):
        output = np.nan_to_num(output, 0.)
        #tempout = output
        minin, maxin = get_aminmax(output)
        output = scale_array_by_amax(output)
    else:
        output = scale_array(output, 0.0, 1.0)

    np.save(filename + ".npy", output)

#############################
#       TRAINING CODE       #
#  DERIVED FROM TENSORFLOW  #
#  AND KERAS DOCUMENTATION  #
#############################

#     https://blog.keras.io/building-autoencoders-in-keras.html     #

def hard_reset():
    encoder = 0
    decoder = 0
    input = 0
    sess = 0
    K.clear_session()
    print("... resetting all models and variables ...")

def import_training_data(input_file):
    global input

    print("... importing training data ...")
    input = np.nan_to_num(np.load(input_file + ".npy"))

def rescale(vector):
    global scale_mult
    global scale_subtract

    return(np.add(np.multiply(vector[0], scale_mult), scale_subtract))

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    global sess

    K.set_session(sess)
    graph = sess.graph
    with graph.as_default():
        z_mean, z_log_var = args
        epsilon = 1e-06
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

     #############################
     #       SHALLOW MODEL       #
     #############################

def init_autoencoder_shallow():

    global intermediate_dim
    global encoder
    global decoder
    global learning_rate
    global vae
    global sess

    K.set_session(sess)
    graph = sess.graph
    with graph.as_default():

        sess = tf.compat.v1.keras.backend.get_session()
        init_op = tf.compat.v1.global_variables_initializer()
        sess.run(init_op)

        original_dim = melsize
        input_shape = (original_dim, )
        latent_dim = encoding_dim

        # VAE model = encoder + decoder
        # build encoder model
        inputs = tf.keras.Input(shape=input_shape, name='encoder_input')
        x = tf.keras.layers.Dense(intermediate_dim, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5), kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-5))(inputs)

        z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
        z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()

        # build decoder model
        latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
        x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = tf.keras.layers.Dense(original_dim, activation='sigmoid')(x)

        # instantiate decoder model
        decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = tf.keras.Model(inputs, outputs, name='vae_mlp')

        reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)

        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        opt = tf.keras.optimizers.Adam(lr=learning_rate)

        vae.compile(optimizer=opt)

     #############################
     #         DEEP MODEL        #
     #   (ARBITRARY STRUCTURE)   #
     #############################

def init_autoencoder_deep():

    global intermediate_dim
    global encoder
    global decoder
    global learning_rate
    global vae
    global sess

    K.set_session(sess)
    graph = sess.graph
    with graph.as_default():

        sess = tf.compat.v1.keras.backend.get_session()
        init_op = tf.compat.v1.global_variables_initializer()
        sess.run(init_op)

        original_dim = melsize

        # network parameters
        input_shape = (original_dim, )
        latent_dim = encoding_dim

        # VAE model = encoder + decoder
        # build encoder model
        inputs = tf.keras.Input(shape=input_shape, name='encoder_input')
        x1 = tf.keras.layers.Dense(intermediate_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                                            bias_regularizer=tf.keras.regularizers.l2(1e-4),
                                                            activity_regularizer=tf.keras.regularizers.l2(1e-5))(inputs)
        x2 = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                                        bias_regularizer=tf.keras.regularizers.l2(1e-4),
                                                        activity_regularizer=tf.keras.regularizers.l2(1e-5))(x1)
        x3 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                                        bias_regularizer=tf.keras.regularizers.l2(1e-4),
                                                        activity_regularizer=tf.keras.regularizers.l2(1e-5))(x2)
        x4 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                                        bias_regularizer=tf.keras.regularizers.l2(1e-4),
                                                        activity_regularizer=tf.keras.regularizers.l2(1e-5))(x3)
        x5 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                                        bias_regularizer=tf.keras.regularizers.l2(1e-4),
                                                        activity_regularizer=tf.keras.regularizers.l2(1e-5))(x4)
        x6 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                                        bias_regularizer=tf.keras.regularizers.l2(1e-4),
                                                        activity_regularizer=tf.keras.regularizers.l2(1e-5))(x5)
        x7 = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                                        bias_regularizer=tf.keras.regularizers.l2(1e-4),
                                                        activity_regularizer=tf.keras.regularizers.l2(1e-5))(x6)
        z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x7)
        z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x7)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()

        # build decoder model
        latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
        x1 = tf.keras.layers.Dense(16, activation='relu')(latent_inputs)
        x2 = tf.keras.layers.Dense(32, activation='relu')(x1)
        x3 = tf.keras.layers.Dense(64, activation='relu')(x2)
        x4 = tf.keras.layers.Dense(128, activation='relu')(x3)
        x5 = tf.keras.layers.Dense(256, activation='relu')(x4)
        x6 = tf.keras.layers.Dense(512, activation='relu')(x5)
        x7 = tf.keras.layers.Dense(intermediate_dim, activation='relu')(x6)
        outputs = tf.keras.layers.Dense(original_dim, activation='sigmoid')(x7)

        # instantiate decoder model
        decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = tf.keras.Model(inputs, outputs, name='vae_mlp')

        reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)

        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        opt = tf.keras.optimizers.Adam(lr=learning_rate)
        vae.compile(optimizer=opt)

     #############################
     #       TRAINING LOOP       #
     #############################

def train(filename):

    global vae
    global input
    global sess

    if(deep == 1):
      print("######################################")
      print("#    SETTING AUTOENCODER TO DEEP     #")
      print("######################################")
      print("")
      init_autoencoder_deep()

    else:
      print("######################################")
      print("#   SETTING AUTOENCODER TO SHALLOW   #")
      print("######################################")
      print("")
      init_autoencoder_shallow()

    K.set_session(sess)
    graph = sess.graph
    with graph.as_default():
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', min_delta=min_delta, patience = regression_patience)
        history = vae.fit(input,
                batch_size = batch_size,
                epochs=n_epochs, verbose = 1, callbacks=[es])

        vae.save_weights(filename + ".h5")
        get_minmax()

#############################
#     SCALING FUNCTIONS     #
#############################

def scale_array(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom

def get_aminmax(X):
    min_ = np.amin(X)
    max_ = np.amax(X)
    return(min_, max_)

def scale_array_by_amax(X):
    '''
    Scale array by total max, not col max
    '''
    return((X - np.amin(X)) / (np.amax(X) - np.amin(X)))

###############################
# MODEL VARIABLE INPUT/OUTPUT #
###############################

def get_minmax():
    global encoder
    global input
    global sess
    global scale_mult
    global scale_subtract

    K.set_session(sess)
    graph = sess.graph
    with graph.as_default():

        z_encoded = encoder.predict(input)
        z_encoded = np.asarray(z_encoded[0], dtype = np.float32)

        min = z_encoded.min(axis = 0)
        max = z_encoded.max(axis = 0)
        scale_mult = np.subtract(max, min)
        scale_subtract = min

def write_minmax(filename):
    global minin
    global maxin
    output_ = np.zeros([1, 2])
    output_[0][0] = minin
    output_[0][1] = maxin
    np.savetxt(filename + ".minmax", output_, delimiter = ", ")#, fmt="%1.6f")

def read_minmax(filename):
    global minin
    global maxin
    input_ = np.loadtxt(filename + ".minmax", delimiter = ", ")
    minin = float(input_[0])
    maxin = float(input_[1])

def write_mm(filename):
    global minin
    global maxin
    global scale_mult
    global scale_subtract
    global intermediate_dim
    global encoding_dim

    output_ = np.zeros([3, encoding_dim])
    output_[0] = scale_mult
    output_[1] = scale_subtract
    output_[2][0] = minin
    output_[2][1] = maxin
    output_[2][2] = intermediate_dim
    output_[2][3] = encoding_dim
    np.savetxt(filename + ".mm", output_, delimiter = ", ", fmt="%1.6f")

def read_mm(filename):
    global minin
    global maxin
    global scale_mult
    global scale_subtract
    input_ = np.loadtxt(filename + ".mm", delimiter = ", ")
    minin = float(input_[2][0])
    maxin = float(input_[2][1])
    scale_mult = input_[0]
    scale_subtract = input_[1]
    intermediate_dim = int(input_[2][2])
    encoding_dim = int(input_[2][3])

def load(filename_):
    global output_frame

    read_mm(filename_)

    output_frame = np.zeros((1, melsize))

    if(deep == 1):
      print("    ######################################")
      print("    #     SETTING AUTOENCODER TO DEEP    #")
      print("    ######################################")
      init_autoencoder_deep()

    else:
      print("    ######################################")
      print("    #    SETTING AUTOENCODER TO SHALLOW  #")
      print("    ######################################")
      init_autoencoder_shallow()

    K.set_session(sess)

    graph = sess.graph
    with graph.as_default():
            vae.load_weights(filename_ + ".h5")
            opt = tf.keras.optimizers.Adam(lr=learning_rate)
            vae.compile(optimizer=opt)

################################
# GENERATE RANDOM PHASE VALUES #
################################
@jit
def gen_phase():
    phase = np.random.rand( 1, 8193) * math.pi * 2.- math.pi
    phase[0, 0] = 0.;
    phase[0, 8192] = 0.;
    return(phase)

################################
#    POST PROCESSING FILTER    #
################################

y1 = 0
y2 = 0
x1 = 0
x2 = 0
a0 = 0.955846
a1 = -1.911692
a2 = 0.955846
b1 = -1.672598
b2 = 0.672771

def post_filter(o):
    global y1
    global y2
    global x1
    global x2

    for i in range(0, o.shape[1]):
        x = o[0, i]
        y = a0 * x + a1 * x1 + a2 * x2 - b1 * y1 - b2 * y2
        y2 = y1
        y1 = y
        x2 = x1
        x1 = x
        o[0, i] = y

    return(o)

def set_brightness():
    global brightness_
    brightness_ = np.zeros((int)(fftsize / 2))
    for i in range(0, (int)(fftsize / 2)):
        brightness_[i] = pow(i/((int)(fftsize / 2)), brightness)

#################################
#     GENERATE INPUT VECTOR     #
#################################
def gen_internal_vector():
    global internal_vector
    # PLACE A USER CONTROLED MAX AND LEVY VALUE HERE
    internal_vector[0,] =  np.clip(np.add(internal_vector, np.divide(np.subtract(np.random.rand(1, 8), .5), random_max_adjusted)), 0., 1.)
#################################
#      DECODE INPUT VECTOR      #
#################################

def decode(*args: List[Any]):
    K.set_session(sess)
    graph = sess.graph
    with graph.as_default():
        encoded = np.zeros((1, 8))
        encoded[0,] = args[0]
        encoded[0,] = rescale(encoded)
        z_sample = [encoded]
        z_out = decoder.predict(z_sample)
        output_frame[0] = z_out
        return(output_frame[0])

#################################
#     AUDIO CALLBACK FOR RT     #
#################################

     ###############################
     #  JIT ACCELERATED FUNCTIONS  #
     ###############################
@jit
def jit_lopas(a,b, f):
    return(np.add(np.multiply(a, f), np.multiply(b, 1.-f)))

@jit
def jit_gainup_and_window(a, w):
    return(np.multiply(np.multiply(a, w),500.))


smooth = np.zeros((512))
#circ_buffer = np.zeros((fftsize))
#circ_buffer_index = 0
co = np.zeros(8193, dtype='complex128')
ca = np.zeros((1, 8193))
cs = np.zeros((fftsize))
sout = np.zeros((fftsize))
se = np.zeros(4096)

def callback(in_data, frame_count, time_info, status):
    global smooth
    #global circ_buffer_index
    global co
    global ca
    global cs
    global sout

    ph = gen_phase()
    gen_internal_vector()
    m = decode(internal_vector[0,])
    #smooth = np.add(np.multiply(m, feedback), np.multiply(smooth, 1. - feedback))
    smooth = jit_lopas(m, smooth, feedback)
    s = mel_to_spectrogram(smooth, mel_inversion_filter, 2.)
    s = np.multiply(s, brightness_)
    s = np.multiply(s, 10)

    ca[0,0:8192] = np.abs(s)
    ca[0,0] = 0.

    co.real = np.multiply(ca, np.cos(ph))
    co.imag = np.multiply(ca, np.sin(ph))
    #cs = np.multiply(np.multiply(scipy.fft.irfft(co), window), 1000.)
    cs = jit_gainup_and_window(scipy.fft.irfft(co), window)

    #print(window.shape)
    sout[0:(fftsize - frame_count)] = sout[frame_count:fftsize]
    sout[(fftsize - frame_count):fftsize] = se
    sout = np.add(sout, cs)[0,]

    #sout = np.add(np.multiply(data, window), sout)[0,]
    #print(np.amax(sout))
    return(sout[0:frame_count].astype(np.float32), pyaudio.paContinue)


#############################
#    OSC MODE FUNCTIONS     #
#############################

def get_decoded(*args: List[Any]):
    K.set_session(sess)
    graph = sess.graph
    with graph.as_default():
        encoded = np.zeros((1, len(args) - 1))
        for i in range (len(args) - 1):
            encoded[0,i] = args[i + 1]

        encoded[0] = rescale(encoded)
        z_sample = [encoded]
        z_out = decoder.predict(z_sample)
        output_frame[0] = z_out
        client.send_message("/decoded", output_frame[0])

def exit_script(arg1):
    os._exit(os.EX_OK)

def ver(arg1):
    if(deep == 0):
        client.send_message("/version", "shallow .1")
    else:
        client.send_message("/version", "deep")

def hard_reset_osc(arg):
    hard_reset()

def set_deep(arg1, arg2):
    global deep
    deep = arg2
    if(deep == 1):
        print("... setting model to deep ...")
    else:
        print("... setting model to shallow ...")

def load_osc(arg1, arg2):
    load(arg2)

#############################
#       INPUT PARSING       #
#############################

if(sys.argv[1] == '-a'):

    print("")
    print("------------------------------")
    print("|         ANALYZING          |")
    print("------------------------------")
    print("")
    filename_ = sys.argv[2]
    start = time.time()
    imported_data = readwave(filename_)
    #imported_data = readwave(filename_)
    prepanalysis(fftsize)
    analyze_data(imported_data, filename_)
    write_minmax(filename_)
    end = time.time()
    print("... time spent analyzing:", end - start)
    print("")

elif(sys.argv[1] == '-t'):

    print("")
    print("------------------------------")
    print("|         TRAINING           |")
    print("------------------------------")
    print("")
    set_batch_size()
    filename_ = sys.argv[2]
    deep = int(sys.argv[3])
    start = time.time()
    import_training_data(filename_)
    read_minmax(filename_)
    end = time.time()
    print("... time spent loading:", end - start)
    print("")
    start = time.time()
    train(filename_)
    write_mm(filename_)
    end = time.time()
    print("")
    print("... time spent training:", end - start, "...")
    print("")

elif(sys.argv[1] == "-r"):


    print("")
    print("------------------------------")
    print("|         RENDERING          |")
    print("------------------------------")
    print("")
    hard_reset()

    filename_ = sys.argv[2]

    load(filename_)
    prepanalysis(fftsize)
    reconstructed = np.zeros((int(sys.argv[3]), fftsize))

    set_brightness()

    o = np.zeros(8193, dtype='complex128')
    a = np.zeros((1, 8193))
    smooth = np.zeros((512))
    for i in range(int(sys.argv[3])):
        ph = gen_phase()
        gen_internal_vector()
        m = decode(internal_vector[0,])
        smooth = np.add(np.multiply(m, feedback), np.multiply(smooth, 1. - feedback))
        s = mel_to_spectrogram(smooth, mel_inversion_filter, 2.)
        s = np.multiply(s, brightness_)
        s = np.multiply(s, 10)
        a[0,0:8192] = np.abs(s)
        a[0,0] = 0.


        o.real = np.multiply(a, np.cos(ph))
        o.imag = np.multiply(a, np.sin(ph))
        reconstructed[i,] = np.multiply(scipy.fft.irfft(o), window)

    output = np.zeros(reconstructed.shape[0] * 4096 + fftsize)

    for i in range(reconstructed.shape[0]):
        #ADD SMOOTHING BEFORE REALTIMING
        output[i * 4096:i * 4096 + fftsize] = np.add(output[i * 4096:i * 4096 + fftsize], reconstructed[i,])

    #o = post_filter(output)

    scipy.io.wavfile.write(sys.argv[2] + "-out.wav", 44100, np.divide(output, np.amax(output)))

elif(sys.argv[1] == "-p"):


    print("")
    print("------------------------------")
    print("|          PLAYING           |")
    print("------------------------------")
    print("")
    hard_reset()


    if 'pyaudio' not in sys.modules:
        print("... pyaudio not found ...")
        print("... exiting ...")
        print("")
        quit()

    filename_ = sys.argv[2]

    load(filename_)
    prepanalysis(fftsize)
    set_brightness()

    p = pyaudio.PyAudio()

    print("... priming model ...")  # THIS IS NEEDED FOR SOME REASON, THE FIRST FEW CALLS TO THE MODEL ARE TOO SLOW OTHERWISE
    for i in range(10):
        m = decode(internal_vector[0,])
    print("... done ...")

    # open stream using callback (3)
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=44100,
                    output=True,
                    frames_per_buffer=4096,
                    stream_callback=callback)

    # start the stream (4)
    stream.start_stream()

    # wait for stream to finish (5)
    while stream.is_active():
        time.sleep(0.1)

    # stop stream (6)
    stream.stop_stream()
    stream.close()

    # close PyAudio (7)
    p.terminate()

elif(sys.argv[1] == "-o"):


    print("")
    print("------------------------------")
    print("|          OSC MODE          |")
    print("------------------------------")
    print("")
    hard_reset()


    client = udp_client.SimpleUDPClient("127.0.0.1", oscinport)

    output_frame =  np.zeros((1, 512))
    #encoded_output_frame = np.zeros((1, 8))

    def test(arg1, arg2):
        print(arg2)

    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/decode", get_decoded)
    dispatcher.map("/load", load_osc)
    dispatcher.map("/quit", exit_script)
    dispatcher.map("/version", ver)
    dispatcher.map("/reset", hard_reset_osc)
    dispatcher.map("/deep", set_deep)
    dispatcher.map("/test", test)


    print("***********************************")
    print("* OPENING CONNECTION ON PORT " + str(oscoutport) + " *")
    print("***********************************")

    server = osc_server.ThreadingOSCUDPServer(
          ("127.0.0.1", oscoutport), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()


elif(sys.argv[1] == "-m"):

    import gc

    print("")
    print("------------------------------")
    print("|       MULTI OSC MODE        |")
    print("------------------------------")
    print("")
    hard_reset()

    client = udp_client.SimpleUDPClient("127.0.0.1", oscinport)

    output_frame =  np.zeros((1, 512))
    #encoded_output_frame = np.zeros((1, 8))

    MODELS = []
    MODELHANDLERS = []
    RANDARR = 0

    gc.disable()


    class mdl:

        scale_mult = 0
        scale_subtract = 0
        epsilon_std = 1.0
        minin = 0.
        maxin = 1.
        vae = 0
        encoder = 0
        decoder = 0
        encoded_output_frame = np.zeros((1, encoding_dim))
        name = ""
        sess = 0



        def __init__(self, input_file):
            K.clear_session()
            self.sess = tf.compat.v1.keras.backend.get_session()
            self.name = input_file
            K.set_session(self.sess)
            graph = self.sess.graph
            with graph.as_default():
                self.load(input_file)

        def load(self, filename):
            print("... loading model ...")

            self.read_mm(filename)

            self.output_frame = np.zeros((1, melsize))
            self.encoded_output_frame = np.zeros((1, encoding_dim))

            if(deep == 1):
              print("######################################")
              print("#    SETTING AUTOENCODER TO DEEP     #")
              print("######################################")
              self.init_autoencoder_deep()
            else:
              print("######################################")
              print("#   SETTING AUTOENCODER TO SHALLOW   #")
              print("######################################")
              self.init_autoencoder_shallow()

            K.set_session(self.sess)
            graph = self.sess.graph
            with graph.as_default():

                self.vae.load_weights(filename + ".h5")
                opt = tf.keras.optimizers.Adam(lr=learning_rate)
                self.vae.compile(optimizer=opt)                #self.get_minmax("")

            print("... done ...")


        def read_mm(self, filename):

            input_ = np.loadtxt(filename + ".mm", delimiter = ", ")
            self.minin = float(input_[2][0])
            self.maxin = float(input_[2][1])
            self.scale_mult = input_[0]
            self.scale_subtract = input_[1]
            #self.intermediate_dim = int(input_[2][2])
            #self.encoding_dim = int(input_[2][3])

        def get_minmax(self, arg1):
            K.set_session(self.sess)
            graph = self.sess.graph
            with graph.as_default():

                z_encoded = self.encoder.predict(self.input)
                z_encoded = np.asarray(z_encoded[0], dtype = np.float32)
                min = z_encoded.min(axis = 0)
                max = z_encoded.max(axis = 0)
                self.scale_mult = np.subtract(max, min)
                self.scale_subtract = min

        def rescale(self, vector):
            return(np.add(np.multiply(vector[0], self.scale_mult), self.scale_subtract))

        def inverse_rescale(self, vector):
            return(np.divide(np.subtract(vector[0], self.scale_subtract), self.scale_mult))

        def sampling(self, args):
            global sess
            K.set_session(sess)
            graph = sess.graph
            with graph.as_default():
                return args[0] + K.exp(0.5 * args[1]) * 1e-06

        def init_autoencoder_shallow(self):

            #self.sess = tf.compat.v1.keras.backend.get_session()
            init_op = tf.compat.v1.global_variables_initializer()
            self.sess.run(init_op)

            self.output_frame = np.zeros((1, melsize))
            self.encoded_output_frame = np.zeros((1, encoding_dim))

            # network parameters
            input_shape = (melsize, )
            #intermediate_dim = int_dim
            latent_dim = encoding_dim

            # VAE model = encoder + decoder
            # build encoder model
            inputs = tf.keras.Input(shape=input_shape, name='encoder_input')
            x = tf.keras.layers.Dense(intermediate_dim, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))(inputs)

            z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
            z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)

            # use reparameterization trick to push the sampling out as input
            # note that "output_shape" isn't necessary with the TensorFlow backend
            z = tf.keras.layers.Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

            # instantiate encoder model
            self.encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
            #self.encoder.summary()

            # build decoder model
            latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
            x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
            outputs = tf.keras.layers.Dense(melsize, activation='sigmoid')(x)

            # instantiate decoder model
            self.decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
            #self.decoder.summary()

            # instantiate VAE model
            outputs = self.decoder(self.encoder(inputs)[2])
            self.vae = tf.keras.Model(inputs, outputs, name='vae_mlp')

            #reconstruction_loss = mse(inputs, outputs)
            reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)

            reconstruction_loss *= melsize
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            vae_loss = K.mean(reconstruction_loss + kl_loss)
            self.vae.add_loss(vae_loss)
            opt = tf.keras.optimizers.Adam(lr=learning_rate)
            self.vae.compile(optimizer=opt)

        def init_autoencoder_deep(self):

            self.sess = tf.compat.v1.keras.backend.get_session()

            # network parameters
            input_shape = (melsize, )
            #intermediate_dim = int_dim
            latent_dim = encoding_dim

            # VAE model = encoder + decoder
            # build encoder model
            inputs = tf.keras.Input(shape=input_shape, name='encoder_input')
            x1 = tf.keras.layers.Dense(intermediate_dim, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))(inputs)
            x2 = tf.keras.layers.Dense(512, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))(x1)
            x3 = tf.keras.layers.Dense(256, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))(x2)
            x4 = tf.keras.layers.Dense(128, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))(x3)
            x5 = tf.keras.layers.Dense(64, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))(x4)
            x6 = tf.keras.layers.Dense(32, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))(x5)
            x7 = tf.keras.layers.Dense(16, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))(x6)
            z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x7)
            z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x7)

            # use reparameterization trick to push the sampling out as input
            # note that "output_shape" isn't necessary with the TensorFlow backend
            z = tf.keras.layers.Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

            # instantiate encoder model
            self.encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
            #self.encoder.summary()

            # build decoder model
            latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
            x1 = tf.keras.layers.Dense(16, activation='relu')(latent_inputs)
            x2 = tf.keras.layers.Dense(32, activation='relu')(x1)
            x3 = tf.keras.layers.Dense(64, activation='relu')(x2)
            x4 = tf.keras.layers.Dense(128, activation='relu')(x3)
            x5 = tf.keras.layers.Dense(256, activation='relu')(x4)
            x6 = tf.keras.layers.Dense(512, activation='relu')(x5)
            x7 = tf.keras.layers.Dense(intermediate_dim, activation='relu')(x6)
            outputs = tf.keras.layers.Dense(melsize, activation='sigmoid')(x7)

            # instantiate decoder model
            self.decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
            #self.decoder.summary()

            # instantiate VAE model
            outputs = decoder(encoder(inputs)[2])
            self.vae = tf.keras.Model(inputs, outputs, name='vae_mlp')

            #reconstruction_loss = mse(inputs, outputs)
            reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)

            reconstruction_loss *= melsize
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            vae_loss = K.mean(reconstruction_loss + kl_loss)
            self.vae.add_loss(vae_loss)
            opt = tf.keras.optimizers.Adam(lr=learning_rate)
            self.vae.compile(optimizer=opt)

        def report(self):
            print("... the model name is", self.name)


    class mdl_handler():

        model_index = 0
        handler_index = 0
        output_frame =  np.zeros((1, melsize))

        def __init__(self, _handler_index, _model_index):
            self.model_index = _model_index
            self.handler_index = _handler_index

        def set(self, _model_index):
            self.model_index = _model_index

        def get_decoded(*args: List[Any]):
            self = args[0]
            #print(self.model_index)
            sess_ = MODELS[self.model_index-1].sess
            K.set_session(sess_) ##### THIS ONE
            graph = sess_.graph
            with graph.as_default():
                encoded = np.zeros((1, len(args[1])))
                for i in range (len(args[1])):
                    encoded[0,i] = args[1][i]
                encoded[0] = MODELS[self.model_index-1].rescale(encoded)
                z_sample = [encoded]
                z_out = MODELS[self.model_index-1].decoder.predict(z_sample)
                self.output_frame[0] = z_out
                client.send_message("/decoded/" + str(self.handler_index), self.output_frame[0])

        def report(self):
            print("... model handler", self.handler_index, "is using model", self.model_index, "...")
            print("... the model name is", MODELS[self.model_index-1].name, "...")

    def multiload(arg1, model_index, filename):
        global MODELS

        if(len(MODELS) <= model_index):
            MODELS.append(mdl(filename))
        else:
            MODELS[model_index - 1] = mdl(filename)


    def loadlist(*args: List[Any]):
        print("HERE")
        n_models = len(args) - 1
        print("... loading", n_models, "models ...")
        for i in range(1, len(args)):
            multiload("",i,args[i])
        print("...........................")
        print("... done loading models ...")
        print("...........................")
        print("")
        generate("", 0.05)
        print("................................")
        print("... done initializing models ...")
        print("................................")

    def setup_handlers(arg1, num):
        global MODELHANDLERS
        global RANDARR

        RANDARR = np.zeros([num, 8])
        MODELHANDLERS = []
        print("... added", num, "model handlers ...")

        for i in range(1, num+1):
            MODELHANDLERS.append(mdl_handler(i, i))

    def set_model(arg1, arg2, arg3):
        MODELHANDLERS[arg2-1].set(arg3)

    def decode(*args: List[Any]):
        index = args[1]
        MODELHANDLERS[index-1].get_decoded(args[2:len(args)])

    def get_num_handlers():
        print("... there are ", len(MODELHANDLERS), "handlers available ..." )

    def get_num_models():
        print("... there are ", len(MODELS), "models loaded ..." )

    def report(arg1):
        get_num_handlers()
        get_num_models()
        for i in range (0, len(MODELHANDLERS)):
            MODELHANDLERS[i].report()
        print("")
        print(".................................")
        print("")
        for i in range (0, len(MODELS)):
            MODELS[i].report()

    def report_single(arg1, model_index):
        print(model_index, MODELS[model_index-1].name)

    def generate(arg1, step):
        global RANDARR
        RND = np.subtract(np.multiply(np.random.rand(len(MODELHANDLERS), 8), (step * 2.)), step)
        RANDARR = np.clip(np.add(RANDARR, RND), 0., 1.)

        for i in range(0, len(MODELHANDLERS)):

            MODELHANDLERS[i].get_decoded(RANDARR[i])

    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/decode", decode)
    dispatcher.map("/setup", setup_handlers)
    dispatcher.map("/set", set_model)
    dispatcher.map("/report", report)
    dispatcher.map("/report/single", report_single)
    dispatcher.map("/load/list", loadlist)
    dispatcher.map("/quit", exit_script)
    dispatcher.map("/version", ver)
    dispatcher.map("/reset", hard_reset)
    dispatcher.map("/deep", set_deep)
    dispatcher.map("/generate", generate)

    print("***********************************")
    print("* OPENING CONNECTION ON PORT " + str(oscoutport) + " *")
    print("***********************************")

    server = osc_server.ThreadingOSCUDPServer(
          ("127.0.0.1", oscoutport), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()
