import os
import argparse
import glob
import pickle

import tensorflow as tf
import numpy as np
from keras_vggface.vggface import VGGFace
from keras.engine import Model
from keras_vggface import utils
from keras.preprocessing import image
from sklearn.decomposition import PCA

LATENT_VARIABLE_OP_NAME = 'fc7'

DEFAULT_WEIGHT = 1.0
FACE_RUNS_FILE_NAME = 'rlf_faces.txt'

'''
Condition 0 is fixation. Condition 1 through X are our latent dimensions. Condition X+1 is one-back.
Condition X+2 through X+22 are the 20 images shown multiple times.
'''
FACE_RUN_FIX_CONDITION = 0
LATENT_DIMENSION = 24
FACE_BIAS_CONDITION_ID = LATENT_DIMENSION + 1
FACE_RUN_ONEBACK_CONDITION = LATENT_DIMENSION + 2
TEST_STIMULI_CONDITION_OFFSET = LATENT_DIMENSION + 3
TEST_STIMULI = {
    'vaegan-sub-01-all': ['M2553.jpg', 'F1631.jpg', 'M2424.jpg', 'F1235.jpg', 'F1148.jpg', 'M2156.jpg', 'F2376.jpg',
                          'M1584.jpg', 'M2466.jpg', 'F2068.jpg', 'F1586.jpg', 'F1232.jpg', 'M2203.jpg', 'M1365.jpg',
                          'M2248.jpg', 'F2467.jpg', 'M2336.jpg', 'F1145.jpg', 'F2377.jpg', 'M2246.jpg'],
    'vaegan-sub-02-all': ['M7260.jpg', 'M8712.jpg', 'M7704.jpg', 'F7216.jpg', 'F8408.jpg', 'F6117.jpg', 'F7792.jpg',
                          'F8669.jpg', 'M4446.jpg', 'M4535.jpg', 'M6338.jpg', 'M7041.jpg', 'F5414.jpg', 'M6776.jpg',
                          'M4621.jpg', 'F4622.jpg', 'F6118.jpg', 'M6953.jpg', 'F5724.jpg', 'F6116.jpg'],
    'vaegan-sub-03-all': ['M10035.jpg', 'F09021.jpg', 'M12366.jpg', 'M10124.jpg', 'M11003.jpg', 'F09903.jpg',
                          'F12323.jpg', 'F10912.jpg', 'M10165.jpg', 'F09152.jpg', 'F11266.jpg', 'M10783.jpg',
                          'F09109.jpg', 'M12233.jpg', 'F11440.jpg', 'M13160.jpg', 'F11400.jpg', 'F08933.jpg',
                          'M08800.jpg', 'M11927.jpg'],
    'vaegan-sub-04-all': ['F14697.jpg', 'F14081.jpg', 'M14039.jpg', 'F13996.jpg', 'M17160.jpg', 'F15049.jpg',
                          'F15137.jpg', 'M13644.jpg', 'M15665.jpg', 'M13289.jpg', 'M15488.jpg', 'M17204.jpg',
                          'F15976.jpg', 'M16368.jpg', 'F13774.jpg', 'F15404.jpg', 'M17336.jpg', 'F13640.jpg',
                          'F14520.jpg', 'M14256.jpg']
}

ALL_TEST_STIMULI = []
for subject_test_stimuli in TEST_STIMULI.values():
    for test_stimuli in subject_test_stimuli:
        ALL_TEST_STIMULI.append(test_stimuli)

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, help='', required=True)
parser.add_argument('--unpackdata_dir', type=str, help='', required=True)
parser.add_argument('--celeba_dir', type=str, help='', required=True)
args = parser.parse_args()


# Layer Features
vgg_model = VGGFace()
out = vgg_model.get_layer(LATENT_VARIABLE_OP_NAME).output
vgg_model_new = Model(vgg_model.input, out)


def encode_img(img_file):
    img = image.load_img(img_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1)  # or version=2
    return vgg_model_new.predict(x)


paradigm_file_name = 'vgg{}'.format(LATENT_VARIABLE_OP_NAME)

subject_nums = range(1, 5)

consolidated_subject_dir_template = 'vaegan-sub-{:02d}-all'

stimulus_to_index = {}
encodings = []
i = 0
for line in open('stimuli/ImageNames2Celeba.txt', 'r'):
    stimulus, celeba = line.split()
    if stimulus in ALL_TEST_STIMULI:
        continue
    celeba_file = os.path.join(args.celeba_dir, celeba)
    latent_values = encode_img(celeba_file).squeeze()
    stimulus_to_index[stimulus] = i
    i += 1
    encodings.append(latent_values)

pca = PCA(n_components=LATENT_DIMENSION)
encodings = np.array(encodings)
pca_encodings = pca.fit_transform(encodings)
with open('pca.pkl', 'wb') as pca_file:
    pickle.dump(pca, pca_file)

for subject_num in subject_nums:
    print('Processing subject {}'.format(subject_num))
    consolidated_subject_name = consolidated_subject_dir_template.format(subject_num)
    consolidated_subject_dir = os.path.join(args.unpackdata_dir, consolidated_subject_name)
    consolidated_subject_bold_dir = os.path.join(consolidated_subject_dir, 'bold')

    test_stimuli = TEST_STIMULI[consolidated_subject_name]
    for face_run in open(os.path.join(consolidated_subject_bold_dir, FACE_RUNS_FILE_NAME), 'r'):
        print('Processing face run {}'.format(face_run))
        face_run_dir = os.path.join(consolidated_subject_bold_dir, '{:03d}'.format(int(face_run)))
        # Convert the events file from OpenNeuro to Freesurfer paradigm format
        paradigm_file = open(os.path.join(face_run_dir, '{}.dyn.para'.format(paradigm_file_name)), 'w')
        event_file = glob.glob(os.path.join(face_run_dir, '*.tsv'))[0]
        onset = 0
        for event_line in open(event_file, 'r'):
            info = event_line.split()
            # Skip the header line and any blank lines
            if 'onset' in info or len(info) == 0:
                continue
            onset = float(info[0])
            duration = float(info[1])
            condition_type = info[2]
            stimulus_filename = info[3]
            base_filename = os.path.basename(stimulus_filename)
            oneback = bool(int(info[4]))

            if base_filename == 'fixation.png':
                condition_id = 0
                paradigm_file.write('{}\t{}\t{}\t{}\t{}\n'.format(
                    onset,
                    condition_id,
                    duration,
                    DEFAULT_WEIGHT,
                    stimulus_filename
                ))
                # Immediately go to the top of the loop so that we don't add the face bias condition at the end of
                # this loop.
                continue
            elif oneback:
                condition_id = FACE_RUN_ONEBACK_CONDITION
                paradigm_file.write('{}\t{}\t{}\t{}\t{}\n'.format(
                    onset,
                    condition_id,
                    duration,
                    DEFAULT_WEIGHT,
                    stimulus_filename
                ))
            elif base_filename in test_stimuli:
                print('found test stimuli')
                condition_id = test_stimuli.index(base_filename) + TEST_STIMULI_CONDITION_OFFSET
                paradigm_file.write('{}\t{}\t{}\t{}\t{}\n'.format(
                    onset,
                    condition_id,
                    duration,
                    DEFAULT_WEIGHT,
                    stimulus_filename
                ))
            else:
                latent_values = pca_encodings[stimulus_to_index[stimulus_filename]]
                assert len(latent_values) == LATENT_DIMENSION

                for index, value in enumerate(latent_values):
                    paradigm_file.write('{}\t{}\t{}\t{}\t{}\n'.format(
                        onset,
                        index + 1,
                        duration,
                        value,
                        stimulus_filename
                    ))

            paradigm_file.write('{}\t{}\t{}\t{}\t{}\n'.format(
                onset,
                FACE_BIAS_CONDITION_ID,
                duration,
                DEFAULT_WEIGHT,
                stimulus_filename
            ))
