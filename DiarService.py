import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from os import path, chdir
import argparse
from magic import magic
from datetime import datetime
import csv
import sys
from wavefile import wavefile

from keras import backend as K
from keras import Input, Model
from keras.layers import Bidirectional, LSTM, Concatenate, BatchNormalization, Dense, GlobalAveragePooling1D, Lambda
from SphereDiar.SphereDiar import SphereDiar


def SphereSpeaker(dimensions=None, num_speak=2000, emb_dim=1000):
    """
    Edited method SphereSpeaker from AM_emb_models in SphereDiar project:
    https://github.com/Livefull/SphereDiar"""
    if dimensions is None:
        dimensions = [59, 201]
    input_feat = Input(shape=(dimensions[1], dimensions[0]))

    # Bidirectional layers (Replaced CuDNNLSTM with LSTM)
    x_1 = Bidirectional(LSTM(250, return_sequences=True))(input_feat)
    x_2 = Bidirectional(LSTM(250, return_sequences=True))(x_1)
    x_3 = Bidirectional(LSTM(250, return_sequences=True))(x_2)

    x_conc = Concatenate(axis=2)([x_1, x_2, x_3])
    emb = BatchNormalization()(x_conc)
    emb = Dense(emb_dim, activation="relu")(emb)
    emb = GlobalAveragePooling1D()(emb)
    emb = BatchNormalization()(emb)
    emb = Lambda(lambda x: K.l2_normalize(x, axis=1))(emb)

    # Softmax layer
    softmax = Dense(num_speak, activation="softmax")(emb)

    test_model = Model(inputs=input_feat, outputs=softmax)
    return test_model


class Settings:
    sample_rate = 16000
    mime_type = 'audio/x-wav'
    frame_len = 2
    hop_len = 0.5
    model_path = path.join(path.dirname(path.abspath(__file__)), "SphereDiar", "models", "SphereSpeaker.hdf")


DO_REPORT = False


def reporting(text, root_step=False):
    if DO_REPORT:
        print((datetime.now().time() if root_step else "\t"), text)


def check_file(filename):
    chdir(path.dirname(filename))
    cur_mime_type = magic.from_file(filename, mime=True)
    if cur_mime_type != Settings.mime_type:
        return False, "The file format is not WAVE audio"

    (rate, sig) = wavefile.load(path.split(filename)[1])

    if sig.shape[0] != 1:
        return False, "The file contains more than one channel (i.e. {}).".format(len(sig))

    if rate != Settings.sample_rate:
        return False, "The file sampling rate is {0} kHz: should be {1} kHz".format(rate / 1000,
                                                                                    Settings.sample_rate / 1000)

    return True, "OK"


def preprocessing(filename):
    """
    Preprocessing and verification of the input file

    :param filename: path to input file
    :raise Exception: if the file format is not suitable
    :return: the signal of input file
    """
    reporting("Preprocessing file...", True)
    chdir(path.dirname(filename))
    (rate, sig) = wavefile.load(path.split(filename)[1])
    signal = sig[0]

    duration = len(signal) / rate
    reporting(f"Done. Duration={duration}")
    return signal


def diarization(signal):
    """
    The basic process of diarization

    :param signal: the signal from input file
    :return: the speaker labels, recognized number of speakers
    """
    try:
        reporting("Diarization...", True)
        SS_model = SphereSpeaker()
        SS_model.load_weights(Settings.model_path)
        SD = SphereDiar(SS_model)
        reporting("The model is loaded.")
        reporting("Feature extraction...")

        SD.extract_features(signal)

        reporting("Getting embeddings...")
        SD.get_embeddings()

        reporting("Clusterization...")
        SD.cluster(rounds=5, debug_info=DO_REPORT)

        reporting(f"Done. Found {SD.opt_speaker_num_} speakers.")
    finally:
        K.clear_session()

    return SD.speaker_labels_, SD.opt_speaker_num_


def lab2seg(labels, frame_len=Settings.frame_len, hop_len=Settings.hop_len):
    """
    Formats a list of labels into an array of named segments.

    :param frame_len: frame duration (in seconds)
    :param labels: a sequence of class labels (per time ``hop_len``)
    :param hop_len: hop length between frames (in seconds)
    :return: a list of segment's limits with the class label.
        `segs[i][0]`, `segs[i][1]` and `segs[i][2]` are start, end point and class label of segment `i`
    """
    reporting("Generation of named segments...", True)

    if len(labels) == 1:
        segs = [0, hop_len, labels[0]]
        return segs

    labels_list = []
    seg_list = []
    ind = 0
    cur_label = labels[ind]
    prev_label = 0
    while ind < len(labels) - 1:
        prev_label = cur_label
        while True:
            ind += 1
            if (labels[ind] != cur_label) | (ind == len(labels) - 1):
                cur_label = labels[ind]
                seg_list.append((ind * hop_len))
                labels_list.append(prev_label)
                break

    if prev_label == cur_label:
        seg_list.append((seg_list.pop() + frame_len))
    else:
        seg_list.append((len(labels) * hop_len))
        labels_list.append(cur_label)

    segs = []
    for i in range(len(seg_list)):
        segs.append([(seg_list[i - 1] if i > 0 else 0.0), seg_list[i], int(labels_list[i])])

    reporting("Done.")
    return segs


def process(filename, debug_mode=False):
    """
    The full process of speaker diarization

    :param debug_mode: print step information if ``debug_mode=True``
    :param filename: path to input file
    :return: the path to csv file with diarization results, recognized number of speakers
    """
    if debug_mode:
        global DO_REPORT
        DO_REPORT = debug_mode

    try:
        signal = preprocessing(filename)
    except BaseException as e:
        print(e)
        sys.exit()

    labels, num_of_speakers = diarization(signal)
    segments = lab2seg(labels)
    res_filename = create_csv(filename, segments)
    return res_filename, num_of_speakers


def create_csv(input_filename, segments, postfix="", headers=None):
    if headers is None:
        headers = ['start_seg', 'end_seg', 'label']
    reporting("Saving file...", True)
    format_file = "." + input_filename.split(".")[-1]
    result_filename = input_filename.replace(format_file, postfix + ".csv")
    with open(result_filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(segments)
    reporting(f"File '{result_filename}' saved. Processing completed.")
    return result_filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Path to input file. The output file will be saved in the same directory.")
    parser.add_argument("-report", action="store_true", help="Enable step-by-step reporting")
    args = parser.parse_args()

    if not path.exists(args.filename):
        print("The file at the specified path does not exist.")
        sys.exit()

    DO_REPORT = args.report

    check_res, msg = check_file(args.filename)
    if check_res:
        process(args.filename)
    else:
        print(msg)
