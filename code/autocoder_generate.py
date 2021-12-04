import sys
import autocoderlib as ac
import numpy as np
import math
import scipy
import pyaudio
import time
import random

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

#internal_vector = 0
#brightness_ = 0
#smooth = 0
#ca = 0
#sout = 0
#se = 0
color = ac.color

if(sys.argv[1] == "-help" or sys.argv[1] == '-h'):
    print()
    print("       python3 ./autocoder_generate.py "+color.CYAN+"-play[-p]"+color.END+" "+color.GREEN+"input_file.wav"+color.END+" fftsize[4096] dac_buffersize[512] ")
    print()
    print("                Generates and plays an output based on the input_file.wav ")
    print("                model via pyAudio.")
    print()
    print("       python3 ./autocoder_generate.py "+color.CYAN+"-render[-r]"+color.END+" "+color.GREEN+"input_file.wav"+color.END+" n fftsize[4096] skip[512]")
    print()
    print("                Generates and saves n windows based on the input_file.wav ")
    print("                model to a file named input_file.wav-out.wav.")
    print()
    print("       python3 ./autocoder_generate.py "+color.CYAN+"-granular[-g]"+color.END+" "+color.GREEN+"input_file.wav"+color.END+" n grainsize trainingskip windowskip rand_n rand_p")
    print()
    print("                Generates and saves n grains based on similarities within")
    print("                the input_file.wav model to a file named input_file.wav-out.wav.")
    print()
    print("                Each new grain is selected from the rand_n most similar")
    print("                candidates. rand_p weights the probability either closer")
    print("                (p > 1.) or further (p < 1.) away from the start point")
    print("                along the similarity metric axis.")
    print()
    print("       python3 ./autocoder_generate.py "+color.CYAN+"-autocode[-a]"+color.END+" "+color.GREEN+"carrier_file.wav modulator_file.wav"+color.END+" fftsize windowskip *args:manipulations")
    print()
    print("                The autocoder takes an input model (carrier_file.wav) and")
    print("                a modulator file (modulator_file.wav), and encodes and  ")
    print("                decodes the modulator using the carrier model. A vector ")
    print("                of the same size as the encoded dimension can be provided")
    print("                to reorder and invert the mapping of each dimension in the")
    print("                input file onto the output model.")
    print()
    exit()

def gen_internal_vector_(random_max, random_pow):
    global internal_vector
    internal_vector[0,] =  np.clip(np.add(internal_vector, np.divide(np.subtract(np.power(np.random.rand(1, 8), random_pow), .5), random_max)), 0., 1.)

def gen_phase(sz_):
    phase = (np.random.rand( 1, int(sz_ / 2) + 1) * math.pi * 2.- math.pi).astype(np.float32)
    phase[0, 0] = 0.;
    phase[0, (int)(sz_ / 2)] = 0.;
    return(phase)

def set_brightness( fftsize, brightness):
    global brightness_
    brightness_ = np.zeros((int)(fftsize / 2), dtype=np.float32)
    for i in range(0, (int)(fftsize / 2)):
        brightness_[i] = pow(i/((int)(fftsize / 2)), brightness)
    brightness_ = np.multiply(brightness_, 10.)

def jit_lopas(a,b, f):
    a = np.multiply(a, f)
    b = np.multiply(b, 1. - f)
    a = np.add(a, b)
    return(a)

def callback(in_data, frame_count, time_info, status):

    global smooth
    global ca
    global brightness_
    global sout

    # DECODE
    gen_internal_vector_(10., 1.)
    p_m = ac.decode(decoder, deep, scale_mult, scale_subtract, internal_vector)
    ca = np.zeros((1, int(fftsize / 2 + 1)), dtype=np.float32)
    ca[0,0:int(fftsize/2)] = p_m.dot(mel_inversion_filter)
    ca = ca.clip(0., 10000.)
    ca[0,0:int(fftsize/2)] = np.abs(np.multiply(ca[0,0:int(fftsize/2)], brightness_))
    ca[0,0] = 0.

    smooth = jit_lopas(ca, smooth, .1)

    # GENERATE NOISE TO USE AS RECONSTRUCTION PHASE
    ph = gen_phase(fftsize)

    # CONVERT BACK TO A SIGNAL
    co = np.zeros(int(fftsize / 2 + 1), dtype='complex64')
    co.real = np.multiply(smooth.real, np.cos(ph))
    co.imag = np.multiply(smooth.imag, np.sin(ph))
    cs = np.multiply(np.fft.irfft(co), window)

    sout[0:(fftsize - frame_count)] = sout[frame_count:fftsize]
    sout[(fftsize - frame_count):fftsize] = se
    sout = np.add(sout, cs)[0,]
    return(np.multiply(sout[0:frame_count].astype(np.float32), 250.), pyaudio.paContinue)

min_a = np.zeros(8)
max_a = np.zeros(8)
min_a.fill(1000000)
max_a.fill(-1000000)

def autocode_norm_factors(in_data):
    global min_a
    global max_a

    amp, ph, norm_factor = ac.analyze_normalized(in_data, window, mel_filter)
    in_data = ac.encode(encoder, deep, scale_mult, scale_subtract, amp)
    t = np.zeros((2, 8))
    t[0,] = in_data
    t[1,] = min_a
    min_a = np.amin(t, axis = 0)
    t[1,] = max_a
    max_a = np.amax(t, axis = 0)



def autocode(in_data, offset, scale, reorder, norm_min, norm_max):

    global smooth
    global ca
    global brightness_

    # DECODE
    amp, ph, norm_factor = ac.analyze_normalized(in_data, window, mel_filter)
    in_data = ac.encode(encoder, deep, scale_mult, scale_subtract, amp)
    in_data = np.divide(np.subtract(in_data,norm_min), np.subtract(norm_max, norm_min))


    in_data = in_data[reorder]
    #in_data = np.add(np.multiply(in_data, invert_mult), invert_add)
    in_data = np.clip(np.add(np.multiply(in_data, scale), offset), 0, 1)
    print(in_data)
    #print(np.add(np.multiply(in_data, scale), offset))
    p_m = np.multiply(ac.decode(decoder, deep, scale_mult, scale_subtract, in_data), norm_factor)
    ca = np.zeros((1, int(fftsize / 2 + 1)), dtype=np.float32)
    ca[0,0:int(fftsize/2)] = p_m.dot(mel_inversion_filter)
    ca = ca.clip(0., 10000.)
    ca[0,0:int(fftsize/2)] = np.abs(np.multiply(ca[0,0:int(fftsize/2)], brightness_))
    ca[0,0] = 0.

    #smooth = jit_lopas(ca, smooth, .1)
    smooth = ca

    # CONVERT BACK TO A SIGNAL
    co = np.zeros(int(fftsize / 2 + 1), dtype='complex64')
    co.real = np.multiply(smooth.real, np.cos(ph))
    co.imag = np.multiply(smooth.imag, np.sin(ph))

    return(np.multiply(np.fft.irfft(co), window))

if(sys.argv[1] == "-play" or sys.argv[1] == "-p"):

    print("")
    print("------------------------------")
    print("|          PLAYING           |")
    print("------------------------------")
    print("")

    minin, maxin, scale_mult, scale_subtract, input_dim, intermediate_dim, encoded_dim, deep = ac.read_mm(sys.argv[2])
    decoder, input_details, output_details = ac.load_lite(sys.argv[2], "decoder")
    print("DEEP:", deep)
    fftsize = int(sys.argv[3])
    buffersize = int(sys.argv[4])
    set_brightness(fftsize, .1)
    mel_filter, mel_inversion_filter, window = ac.initialize(fftsize, input_dim)

    ### INITIALIZE GLOBAL BUFFERS, TRY TO GET RID OF THESE
    smooth = np.zeros((1, int(fftsize / 2 + 1)), dtype=np.float32)
    internal_vector = np.zeros((1, encoded_dim))
    sout = np.zeros((fftsize), dtype=np.float32)
    se = np.zeros(buffersize, dtype=np.float32)

    p = pyaudio.PyAudio()

    # open stream using callback (3)
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=44100,
                    output=True,
                    frames_per_buffer=buffersize,
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

elif(sys.argv[1] == "-render" or sys.argv[1] == "-r"):

    print("")
    print("------------------------------")
    print("|         RENDERING          |")
    print("------------------------------")
    print("")



    filename_ = sys.argv[2]
    n = int(sys.argv[3])
    fftsize = int(sys.argv[4])
    skip = int(sys.argv[5])

    minin, maxin, scale_mult, scale_subtract, input_dim, intermediate_dim, encoded_dim, deep = ac.read_mm(sys.argv[2])
    decoder, input_details, output_details = ac.load_lite(sys.argv[2], "decoder")
    set_brightness(fftsize, .1)
    mel_filter, mel_inversion_filter, window = ac.initialize(fftsize, input_dim)

    ### INITIALIZE GLOBAL BUFFERS, TRY TO GET RID OF THESE
    output = np.zeros(n * skip)
    smooth = np.zeros((1, int(fftsize / 2 + 1)), dtype=np.float32)
    internal_vector = np.zeros((1, encoded_dim))
    sout = np.zeros((fftsize), dtype=np.float32)
    se = np.zeros(skip, dtype=np.float32)

    for i in range(n):
        output[i * skip:(i + 1) * skip], paval = callback(0, skip, 0, 0)

    scipy.io.wavfile.write(sys.argv[2] + "-out.wav", 44100, np.divide(output, np.amax(output)))


elif(sys.argv[1] == "-granular" or sys.argv[1] == "-g"):

    print("")
    print("------------------------------")
    print("| ORDERED GRANULAR RENDERING |")
    print("------------------------------")
    print("")

    filename_ = sys.argv[2]
    n = int(sys.argv[3])
    grainsize = int(sys.argv[4])
    trainingskip = int(sys.argv[5])   ### GET THIS FROM THE MM FILE,
    windowskip = int(sys.argv[6])
    max_rand_step = float(sys.argv[7])
    rand_pow =  float(sys.argv[8])

    minin, maxin, scale_mult, scale_subtract, input_dim, intermediate_dim, encoded_dim, deep = ac.read_mm(sys.argv[2])
    decoder, input_details, output_details = ac.load_lite(sys.argv[2], "decoder")
    set_brightness(grainsize, .1)
    mel_filter, mel_inversion_filter, window = ac.initialize(grainsize, input_dim)

    output = np.zeros(n * windowskip)

    # READ THE WAVEFILE
    wavefile = ac.readwave(filename_)
    print("   WAVEFILE LENGTH:", wavefile.shape)

    # LOAD THE ORDER FILE
    order =  np.load(filename_ + ".ord.npy")
    print("   ORDER FILE SHAPE:", order.shape)

    reconstructed = np.zeros((n, grainsize))

    index_ =  np.random.randint(0, order.shape[0])

    for i in range(n):
        # 2. GET THE GRAIN AND WINDOW IT
        reconstructed[i,] = np.multiply(wavefile[(index_ * trainingskip):((index_ * trainingskip)+ grainsize)], window)

        # 3. RANDOM WALK
        index_ = order[index_, int(pow(random.random(), rand_pow) * max_rand_step)]

    output = np.zeros(reconstructed.shape[0] * windowskip + grainsize)

    for i in range(reconstructed.shape[0]):
        output[i * windowskip:i * windowskip + grainsize] = np.add(output[i * windowskip:i * windowskip + grainsize], reconstructed[i,])

    scipy.io.wavfile.write(sys.argv[2] + "-out.wav", 44100, np.divide(output, np.amax(output)))

elif(sys.argv[1] == "-autocode" or sys.argv[1] == "-a"):

    print("")
    print("------------------------------")
    print("|         AUTOCODING         |")
    print("------------------------------")
    print("")


    model_filename = sys.argv[2]
    input_filename = sys.argv[3]
    fftsize = int(sys.argv[4])
    windowskip = int(sys.argv[5])
    offset = 0 #float(sys.argv[6])
    scale = 1 #float(sys.argv[7])

    minin, maxin, scale_mult, scale_subtract, input_dim, intermediate_dim, encoded_dim, deep = ac.read_mm(model_filename)
    decoder, input_details, output_details = ac.load_lite(model_filename, "decoder")
    encoder, input_details, output_details = ac.load_lite(model_filename, "encoder")
    set_brightness(fftsize, .1)
    mel_filter, mel_inversion_filter, window = ac.initialize(fftsize, input_dim)

    reorder = np.zeros((encoded_dim), dtype = np.int)
    scale = np.zeros((encoded_dim), dtype = np.float32)
    offset = np.zeros((encoded_dim), dtype = np.float32)


    for i in range(0, encoded_dim):
        reorder[i] = int(sys.argv[6 + i])
        scale[i] = float(sys.argv[6 + encoded_dim + i])
        offset[i] = float(sys.argv[6 + encoded_dim + encoded_dim + i])


    #reorder = np.subtract(np.abs(reorder), 1)

    # READ THE WAVEFILE
    wavefile = ac.readwave(input_filename)

    n = int((wavefile.shape[0] - fftsize) / windowskip)

    for i in range(n):
        # 2. GET THE GRAIN AND WINDOW IT
        autocode_norm_factors(np.multiply(wavefile[(i * windowskip):((i * windowskip)+ fftsize)], window))

    #np.divide(np.subtract(a, np.amin(a,axis=0)), np.subtract(np.amax(a, axis = 0), np.amin(a, axis = 0)))
    #quit()

    reconstructed = np.zeros((n, fftsize))

    # NORMALIZE THE INPUT
    for i in range(n):
        # 2. GET THE GRAIN AND WINDOW IT
        reconstructed[i,] = autocode(np.multiply(wavefile[(i * windowskip):((i * windowskip)+ fftsize)], window), offset, scale, reorder, min_a, max_a)


    output = np.zeros(reconstructed.shape[0] * windowskip + fftsize)

    for i in range(reconstructed.shape[0]):
        output[i * windowskip:i * windowskip + fftsize] = np.add(output[i * windowskip:i * windowskip + fftsize], reconstructed[i,])

    scipy.io.wavfile.write(sys.argv[2] + "-out.wav", 44100, np.divide(output, np.amax(output)))
