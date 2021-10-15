import os
import argparse
import glob

import tensorflow as tf
import numpy as np

from disentanglement_lib.data.ground_truth.celeba import process_path

LATENT_VARIABLE_OP_NAME = 'encoder/means/BiasAdd:0'

DEFAULT_WEIGHT = 1.0
LOCALIZER_RUNS_FILE_NAME = 'rlf_localizer.txt'
FACE_RUNS_FILE_NAME = 'rlf_faces.txt'
LOCALIZER_CONDITION_IDS = {
    'fix': 0,
    'face': 1,
    'object': 2
}

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


def encode_img(img_file):
    input_image = process_path(img_file, size=(64, 64))
    # We need to add an extra dimension for a batch size of 1
    input_image = np.expand_dims(input_image, axis=0)
    latent_variable = sess.run(LATENT_VARIABLE_OP_NAME, feed_dict={'Placeholder:0': input_image})
    # The first index is the batch of size 1 so index into that
    return latent_variable[0]


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, help='', required=True)
parser.add_argument('--unpackdata_dir', type=str, help='', required=True)
parser.add_argument('--celeba_dir', type=str, help='', required=True)
parser.add_argument('--do_localizer', action='store_true', required=False)
parser.add_argument('--tag', type=str, default='', help='A tag to add to the saved model name')
args = parser.parse_args()

saved_model_dir = os.path.join('output', args.model_dir, 'tfhub')
sess = tf.Session()
model = tf.saved_model.load(export_dir=saved_model_dir, tags=[], sess=sess)

paradigm_file_name = args.model_dir.replace('/', '.')

subject_nums = range(1, 5)

consolidated_subject_dir_template = 'vaegan-sub-{:02d}-all'

stimulus_to_celeba = {}
for line in open('stimuli/ImageNames2Celeba.txt', 'r'):
    stimulus, celeba = line.split()
    stimulus_to_celeba[stimulus] = celeba

for subject_num in subject_nums:
    print('Processing subject {}'.format(subject_num))
    consolidated_subject_name = consolidated_subject_dir_template.format(subject_num)
    consolidated_subject_dir = os.path.join(args.unpackdata_dir, consolidated_subject_name)
    consolidated_subject_bold_dir = os.path.join(consolidated_subject_dir, 'bold')

    # Create the localizer paradigm files
    if args.do_localizer:
        for localizer_run in open(os.path.join(consolidated_subject_bold_dir, LOCALIZER_RUNS_FILE_NAME), 'r'):
            localizer_run_dir = os.path.join(consolidated_subject_bold_dir, '{:03d}'.format(int(localizer_run)))
            # Convert the events file from OpenNeuro to Freesurfer paradigm format
            paradigm_file = open(os.path.join(localizer_run_dir, 'localizer.dyn.para'), 'w')
            event_file = glob.glob(os.path.join(localizer_run_dir, '*.tsv'))[0]
            onset = 0
            for event_line in open(event_file, 'r'):
                info = event_line.split()
                # Skip the header line
                if 'onset' in info:
                    continue
                duration = int(info[1])
                condition_type = info[2]
                condition_id = LOCALIZER_CONDITION_IDS[condition_type]
                paradigm_file.write('{}\t{}\t{}\t{}\t{}\n'.format(
                    onset,
                    condition_id,
                    duration,
                    DEFAULT_WEIGHT,
                    condition_type
                ))
                onset += duration

    test_stimuli = TEST_STIMULI[consolidated_subject_name]
    for face_run in open(os.path.join(consolidated_subject_bold_dir, FACE_RUNS_FILE_NAME), 'r'):
        print('Processing face run {}'.format(face_run))
        face_run_dir = os.path.join(consolidated_subject_bold_dir, '{:03d}'.format(int(face_run)))
        # Convert the events file from OpenNeuro to Freesurfer paradigm format
        if args.tag:
            paradigm_file = open(os.path.join(face_run_dir, '{}.{}.dyn.para'.format(args.tag, paradigm_file_name)), 'w')
        else:
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
                # Encode the stimulus in the VAE
                celeba_file = stimulus_to_celeba[stimulus_filename]
                celeba_file = os.path.join(args.celeba_dir, celeba_file)
                latent_values = encode_img(celeba_file)
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