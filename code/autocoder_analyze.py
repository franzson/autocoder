# DOCUMENT ALL
# BUILD THE FRAMEWORK AROUND THE FILE DISTANCE STUFF
# TEST THE DISTANCE STUFF
# SAVE A COPY OF THE COLAB ALONG WITH THE MASTER CODE

import sys
import autocoderlib as ac
import numpy as np
import os
import scipy

fftsize = 16384
windowskip = 1024

regression_patience = 1000
learning_rate = .00001
min_delta = .00001

input_dim = 512
intermediate_dim = 1000
encoded_dim = 8
color = ac.color

if(len(sys.argv) < 2):
    exit()

if(sys.argv[1] == '-help' or sys.argv[1] == '-h'):
    print()
    print()
    print("                         ... AUTOCODER ANALYZE ...")
    print()
    print("       python3 ./autocoder_analyze.py "+color.CYAN+"-analyze[-a]"+color.END+" "+color.GREEN+"input_file.wav"+color.END+"")
    print()
    print("                Segments an input file into windows and runs a mel")
    print("                scale conversion to create a training data set.")
    print()
    print("                Produces two files, one containing the training set")
    print("                (input_sound.wav.npy), the other containing the normal-")
    print("                ization factors for the dataset (input_sound.wav.minmax).")
    print()
    print("       python3 ./autocoder_analyze.py "+color.CYAN+"-analyze[-a]"+color.END+" "+color.GREEN+"/input_folder"+color.END+" n_windows")
    print()
    print("                Segments each file in /input_folder into windows, ")
    print("                samples n_windows from the dataset, and runs a mel")
    print("                scale conversion to create a training data set.")
    print()
    print("                Produces three files, one containing the training set")
    print("                (combined.wav.npy), the second containing the normal-")
    print("                ization factors for the dataset (combined.wav.minmax),")
    print("                and a stand-in wave file for other functions")
    print("                (combined.wav).")
    print()
    print("       python3 ./autocoder_analyze.py "+color.CYAN+"-analyze[-a]"+color.END+" "+color.GREEN+"/input_folder"+color.END+" average")
    print()
    print("                Same as analyze-folder except the frames of each file in")
    print("                /input_folder are averaged together as a single entry")
    print("                in the dataset (for use in similarity calculations).")
    print()
    print("       python3 ./autocoder_analyze.py "+color.CYAN+"-analyze[-a]"+color.END+" "+color.GREEN+"/input_folder"+color.END+" first")
    print()
    print("                Same as analyze-folder except only the first frame of")
    print("                each file in /input_folder is used in the dataset.")
    print()
    print("       python3 ./autocoder_analyze.py "+color.CYAN+"-analyze_normalized[-an]"+color.END+" "+color.GREEN+"input_file.wav"+color.END+"")
    print()
    print("                Same as analyze-file except each frame of the sound")
    print("                is normalized independently (for use with autocoding).")
    print()
    print()
    print("                           ... AUTOCODER TRAIN ...")
    print("")
    print("       python3 ./autocoder_analyze.py "+color.CYAN+"-train[-t]"+color.END+" "+color.GREEN+"input_file.wav"+color.END+" deep[0/1] ")
    print("       "+color.YELLOW+"(regression_patience[1/n] learning_rate[.00001] min_delta[.00001])"+color.END+" ")
    print()
    print("                Trains a variational autoencoder on a data set for")
    print("                input_file locally. The deep parameter selects between")
    print("                a single layer model (very efficient) and a 7 layer")
    print("                deep model (vanishing returns on the deep model).")
    print()
    print("                Regression patience--ie how often the training is ")
    print("                allowed to regress, the learning rate--ie how much")
    print("                the model can change on each update, and the minimum")
    print("                delta––the minimum difference between the training")
    print("                error on successive runs that should be reached before")
    print("                exiting training, can be adjusted in the input string.")
    print()
    print("                This is very slow and should be done on google collab.")
    print()
    print()
    print("                         ... ENCODE / DISTANCE ...")
    print("")
    print("       python3 ./autocoder_analyze.py "+color.CYAN+"-encode[-e]"+color.END+" "+color.GREEN+"input_file.wav"+color.END+"")
    print()
    print("                Encodes each data point in the dataset associated")
    print("                with input-file using the encoder from the variational")
    print("                autoencoder trained on the same data set.")
    print()
    print("                This allows distance calculations to be performed")
    print("                across the training set.")
    print()
    print("       python3 ./autocoder_analyze.py "+color.CYAN+"-distances[-d]"+color.END+" "+color.GREEN+"input_file.wav"+color.END+" n_most_similar")
    print()
    print("                Returns a matrix containing the index of the n most")
    print("                similar frames for each frame in the dataset.")
    print()
    print()
    print("                            ... SANITY CHECK ...")
    print()
    print("       python3 ./autocoder_analyze.py "+color.CYAN+"-test_decode/-test_encode/-test_h5"+color.END+" "+color.GREEN+"input_file.wav"+color.END)
    print()
    print("                Functions to test if Tensorflow is working correctly.")
    print()
    print("       For simplicity, input_file.wav is treated as an anchor to load")
    print("       various other files generated by the other functions.")
    print()
    exit()


elif(sys.argv[1] == '-analyze' or sys.argv[1] == '-a'):

    print()
    print("------------------------------")
    print("|          ANALYZE           |")
    print("------------------------------")
    print()
    filename = sys.argv[2]
    averaging = False
    first_frame = False
    if(os.path.isdir(filename)):
        if(os.path.exists(os.path.join(sys.argv[2], "combined.wav"))):
            os.remove(os.path.join(sys.argv[2], "combined.wav"))
        n_files = 0
        if(sys.argv[3] == 'average'):
            averaging = True
        elif(sys.argv[3] == 'first'):
            first_frame = True
        else:
            n_frames = int(sys.argv[3]) #50


        for file in os.listdir(sys.argv[2]):
            if file.endswith(".wav"):
                print(file)
                filename_ = os.path.join(sys.argv[2], file)
                imported_data = ac.readwave(filename_)
                mel_filter, mel_inversion_filter, window = ac.initialize(fftsize, input_dim)
                minin, maxin = ac.analyze_data(imported_data, filename_, fftsize, windowskip, input_dim, window, mel_filter)
                ac.write_minmax(filename_, minin, maxin)
                n_files = n_files + 1

        n = 0
        output = 0
        filenames = np.zeros(n_files, dtype = object)
        if(averaging != True and first_frame != True):
            output = np.zeros((n_files * n_frames, input_dim))
            for file in os.listdir(sys.argv[2]):
                if file.endswith(".wav"):
                    filenames[n] = file
                    filename_ = os.path.join(sys.argv[2], file)
                    imported_data = ac.import_training_data(filename_)
                    frames = imported_data[np.random.randint(0, imported_data.shape[0], n_frames)]
                    output[n:(n + n_frames),] = frames
                    n = n + 1
                    os.remove(filename_ + ".npy")
                    os.remove(filename_ + ".minmax")
        elif (averaging == True):
            output = np.zeros((n_files, input_dim))
            for file in os.listdir(sys.argv[2]):
                if file.endswith(".wav"):
                    filenames[n] = file
                    filename_ = os.path.join(sys.argv[2], file)
                    imported_data = ac.import_training_data(filename_)
                    output[n,] = np.sum(imported_data, axis = 0) / imported_data.shape[0]
                    n = n + 1
                    os.remove(filename_ + ".npy")
                    os.remove(filename_ + ".minmax")
        elif (first_frame == True):
            output = np.zeros((n_files, input_dim))
            for file in os.listdir(sys.argv[2]):
                if file.endswith(".wav"):
                    print(file)
                    filenames[n] = file
                    filename_ = os.path.join(sys.argv[2], file)
                    imported_data = ac.import_training_data(filename_)
                    output[n,] = imported_data[0,]
                    n = n + 1
                    os.remove(filename_ + ".npy")
                    os.remove(filename_ + ".minmax")
        print(filenames)
        output = ac.scale_array_by_amax(output)
        filename_ = os.path.join(sys.argv[2], "combined.wav")
        f = open(filename_, "w")
        f.close()
        ac.write_minmax(filename_, minin, maxin)
        np.save(filename_ + ".npy", output)
        f = open(filename_ + ".txt", "w")
        for i in range(0, filenames.shape[0]):
            f.write(str(i) + ", " + filenames[i] + ";")
        f.close()
    else:
        imported_data = ac.readwave(filename)
        mel_filter, mel_inversion_filter, window = ac.initialize(fftsize, input_dim)
        minin, maxin = ac.analyze_data(imported_data, filename, fftsize, windowskip, input_dim, window, mel_filter)
        ac.write_minmax(filename, minin, maxin)

if(sys.argv[1] == '-analyze_normalized' or sys.argv[1] == '-an'):

    print()
    print("------------------------------")
    print("|     ANALYZE NORMALIZED     |")
    print("------------------------------")
    print()
    filename_ = sys.argv[2]
    imported_data = ac.readwave(filename_)
    mel_filter, mel_inversion_filter, window = ac.initialize(fftsize, input_dim)
    minin, maxin = ac.analyze_data_normalized(imported_data, filename_, fftsize, windowskip, input_dim, window, mel_filter)
    ac.write_minmax(filename_, minin, maxin)

elif(sys.argv[1] == '-t'):

    print()
    print("------------------------------")
    print("|         TRAINING           |")
    print("------------------------------")
    print()

    if(len(sys.argv) > 4):
        regression_patience = int(sys.argv[4])

    if(len(sys.argv) > 6):
        learning_rate = float(sys.argv[5])
        min_delta = float(sys.argv[6])

    filename_ = sys.argv[2]
    deep = int(sys.argv[3])
    input_data = ac.import_training_data(filename_)
    minin, maxin = ac.read_minmax(filename_)

    if(deep == 0):
        vae, encoder, decoder = ac.init_autoencoder_shallow(input_dim, intermediate_dim, encoded_dim, learning_rate)
    else:
        vae, encoder, decoder = ac.init_autoencoder_deep(input_dim, intermediate_dim, encoded_dim, learning_rate)

    vae, encoder, decoder, scale_mult, scale_subtract = ac.train(filename_, vae, encoder, decoder, input_data, min_delta, regression_patience, ac.get_batch_size(), deep)

    ac.write_mm(filename_, minin, maxin, scale_mult, scale_subtract, input_dim, intermediate_dim, encoded_dim, deep)

elif(sys.argv[1] == '-encode' or sys.argv[1] == '-e'):

    ''' ENCODES EACH DATA POINT IN THE INPUT SET TO ALLOW DISTANCE CALCULATIONS
        ACROSS THE SET'''


    print()
    print("------------------------------")
    print("|   ENCODE A TRAINING SET    |")
    print("------------------------------")
    print()

    filename_ = sys.argv[2]
    input_data = ac.import_training_data(filename_)

    minin, maxin, scale_mult, scale_subtract, input_dim, intermediate_dim, encoded_dim, deep = ac.read_mm(sys.argv[2])
    encoder, input_details, output_details = ac.load_lite(filename_, "encoder")

    encoded_input = np.zeros((input_data.shape[0], encoded_dim))

    for i in range(0, encoded_input.shape[0]):
        encoded_input[i,] = ac.encode(encoder, deep, scale_mult, scale_subtract, input_data[i,])

    np.save(filename_ + ".enc", encoded_input)

elif(sys.argv[1] == '-distances' or sys.argv[1] == '-d'):

    ''' RETURNS A MATRIX OF THE sys.argv[3] MOST SIMILAR DATA POINTS FOR EACH
        DATA POINT IN THE INPUT. THIS IS USEFUL FOR SIMILARITY JUDGEMENTS OR
        GRANULAR SYNTHESIS
        -decode[-d] file n_to_keep        '''

    print()
    print("------------------------------")
    print("|    CALCULATE DISTANCES     |")
    print("------------------------------")
    print()

    filename_ = sys.argv[2]
    group = False
    group_n = 1

    if(len(sys.argv) > 4):
        group = True
        group_n = int(sys.argv[4])

    encoded_input = np.load(filename_ + ".enc.npy")

    if(group):
        t = np.zeros((int(encoded_input.shape[0]/group_n), encoded_input.shape[1]))
        sd = np.zeros((int(encoded_input.shape[0]/group_n), encoded_input.shape[1]))
        for i in range(0, int(encoded_input.shape[0]/group_n)):
            t[i,] = np.sum(encoded_input[i * group_n:(i + 1) * group_n,], axis = 0) / group_n
            sd[i,] = np.std(encoded_input[i * group_n:(i + 1) * group_n,], axis = 0)
        encoded_input = t

    distances = np.zeros((encoded_input.shape[0], encoded_input.shape[0]))

    for i in range(0, distances.shape[0]):
        for j in range(0, distances.shape[0]):
            if(group):
                distances[i,j] = np.sum(np.multiply(np.abs(np.subtract(t[i,], t[j,])), np.subtract(1, sd[j,])))
            else:
                distances[i,j] = np.sum(np.abs(np.subtract(encoded_input[i,], encoded_input[j,])))

    n_returns = min([int(sys.argv[3]), distances.shape[0] - 1])

    if(group):
        order = np.zeros((t.shape[0], n_returns), dtype='int')
        for i in range(0, t.shape[0]):
            order[i,] = np.argsort(distances[i,])[1:n_returns + 1]
    else:
        order = np.zeros((encoded_input.shape[0], n_returns), dtype='int')
        for i in range(0, encoded_input.shape[0]):
            order[i,] = np.argsort(distances[i,])[1:n_returns + 1]
    np.save(filename_ + ".ord", order)
    if(os.path.exists(filename_+ ".txt")):
        f = open(filename_ + ".txt")
        lines = f.readlines()

        print("    .......... files in similarity order ..........")
        print()
        for i in range(0, len(lines)):
            print("   ", lines[i][0:-2] + ":    ", end=" ")
            for j in range(0, order.shape[1]):
                print(lines[order[i,j]][0:-2], end=" ")
            print()
            print()

elif(sys.argv[1] == '-test_encode'):
    ###  ENCODE VECTOR (TFLITE)

    minin, maxin, scale_mult, scale_subtract, input_dim, intermediate_dim, encoded_dim, deep = ac.read_mm(sys.argv[2])

    encoder, input_details, output_details = ac.load_lite(sys.argv[2], "encoder")

    in_vect = np.zeros(input_dim)
    print(ac.encode(encoder, int(sys.argv[3]), scale_mult, scale_subtract, in_vect))

elif(sys.argv[1] == '-test_decode'):
    ###  DECODE VECTOR (TFLITE)

    minin, maxin, scale_mult, scale_subtract, input_dim, intermediate_dim, encoded_dim, deep = ac.read_mm(sys.argv[2])

    decoder, input_details, output_details = ac.load_lite(sys.argv[2], "decoder")

    in_vect = np.zeros(encoded_dim)
    print(ac.decode(decoder, deep, scale_mult, scale_subtract, in_vect))

elif(sys.argv[1] == '-test_h5'):
    ###  ENCODE/DECODE VECTOR (FULL)
    ### TEST ON INTEL

    minin, maxin, scale_mult, scale_subtract, input_dim, intermediate_dim, encoded_dim, deep = ac.read_mm(sys.argv[2])

    if(deep == 0):
        vae, encoder, decoder =  ac.init_autoencoder_shallow(input_dim, intermediate_dim, encoded_dim, .0001)
    else:
        vae, encoder, decoder =  ac.init_autoencoder_deep(input_dim, intermediate_dim, encoded_dim, .0001)

    vae, encoder, decoder = ac.load(sys.argv[2], vae, encoder, decoder)

    in_vect = np.zeros(encoded_dim)
    print(ac.decode(decoder, deep, scale_mult, scale_subtract, in_vect))

    in_vect = np.zeros(input_dim)
    print(ac.encode(encoder, deep, scale_mult, scale_subtract, in_vect))
