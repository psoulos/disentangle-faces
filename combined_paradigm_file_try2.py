import os
import argparse


DEFAULT_WEIGHT = 1.0
FACE_RUNS_FILE_NAME = 'rlf_faces.txt'

parser = argparse.ArgumentParser()
parser.add_argument('--model_one_name', type=str, help='', required=True)
parser.add_argument('--model_two_name', type=str, help='', required=True)
parser.add_argument('--model_one_dimensions', type=int, help='', required=True)
parser.add_argument('--model_two_dimensions', type=int, help='', required=True)
parser.add_argument('--unpackdata_dir', type=str, help='', required=True)
args = parser.parse_args()

model_name = 'combined.{}.{}'.format(args.model_one_name, args.model_two_name)
'''
Condition 0 is fixation. Condition 1 through X are our latent dimensions. Condition X+1 is one-back.
Condition X+2 through X+22 are the 20 images shown multiple times.
'''
FACE_RUN_FIX_CONDITION = 0
LATENT_DIMENSION = args.model_one_dimensions + args.model_two_dimensions
DEFAULT_WEIGHT = 1.0

# Used for face bias
FACE_BIAS_CONDITION_ID = LATENT_DIMENSION + 1
FACE_RUN_ONEBACK_CONDITION = LATENT_DIMENSION + 2
TEST_STIMULI_CONDITION_OFFSET = LATENT_DIMENSION + 3

subject_nums = range(1, 5)

consolidated_subject_dir_template = 'vaegan-sub-{:02d}-all'

PARADIGM_LINE_ONSET_INDEX = 0
PARADIGM_LINE_CONDITION_INDEX = 1
PARADIGM_LINE_DURATION_INDEX = 2
PARADIGM_LINE_VALUE_INDEX = 3
PARADIGM_LINE_FILENAME_INDEX = 4


def extract_condition_and_value(line):
    split_line = line.split('\t')
    return split_line[PARADIGM_LINE_CONDITION_INDEX], split_line[PARADIGM_LINE_VALUE_INDEX]


for subject_num in subject_nums:
    print('Processing subject {}'.format(subject_num))
    consolidated_subject_name = consolidated_subject_dir_template.format(subject_num)
    consolidated_subject_dir = os.path.join(args.unpackdata_dir, consolidated_subject_name)
    consolidated_subject_bold_dir = os.path.join(consolidated_subject_dir, 'bold')

    for face_run in open(os.path.join(consolidated_subject_bold_dir, FACE_RUNS_FILE_NAME), 'r'):
        print('Processing face run {}'.format(face_run))
        face_run_dir = os.path.join(consolidated_subject_bold_dir, '{:03d}'.format(int(face_run)))
        # Convert the events file from OpenNeuro to Freesurfer paradigm format
        paradigm_file = open(os.path.join(face_run_dir, '{}.dyn.para'.format(model_name)), 'w')

        model_one_paradigm = '{}.dyn.para'.format(args.model_one_name)
        model_two_paradigm = '{}.dyn.para'.format(args.model_two_name)
        model_one_file = open(os.path.join(face_run_dir, model_one_paradigm), 'r')
        model_two_file = open(os.path.join(face_run_dir, model_two_paradigm), 'r')

        model_one_output = []
        model_two_output = []
        for line in model_one_file:
            model_one_output.append(line)
        for line in model_two_file:
            model_two_output.append(line)

        model_one_current_line = 0
        model_two_current_line = 0
        while model_one_current_line < len(model_one_output):
            current_line = model_one_output[model_one_current_line].split('\t')
            if int(current_line[PARADIGM_LINE_CONDITION_INDEX]) == 0:
                paradigm_file.write('{}\t{}\t{}\t{}\t{}'.format(
                    current_line[PARADIGM_LINE_ONSET_INDEX],
                    0,
                    current_line[PARADIGM_LINE_DURATION_INDEX],
                    DEFAULT_WEIGHT,
                    current_line[PARADIGM_LINE_FILENAME_INDEX]
                ))
                model_one_current_line += 1
                model_two_current_line += 1
                continue
            # If we are in the range of oneback or test stimuli, just write that line and continue
            elif int(current_line[PARADIGM_LINE_CONDITION_INDEX]) >= args.model_one_dimensions + 2:
                paradigm_file.write('{}\t{}\t{}\t{}\t{}'.format(
                    current_line[PARADIGM_LINE_ONSET_INDEX],
                    # Off set the condition ids to account for the increased dimensionality owing to model two
                    int(current_line[PARADIGM_LINE_CONDITION_INDEX]) + args.model_two_dimensions,
                    current_line[PARADIGM_LINE_DURATION_INDEX],
                    DEFAULT_WEIGHT,
                    current_line[PARADIGM_LINE_FILENAME_INDEX]
                ))
                model_one_current_line += 1
                model_two_current_line += 1
                continue
            # This is a standard training face image
            onset = current_line[PARADIGM_LINE_ONSET_INDEX]
            duration = current_line[PARADIGM_LINE_DURATION_INDEX]
            # Note the stimulus filename ends in \n
            stimulus_filename = current_line[PARADIGM_LINE_FILENAME_INDEX]
            for condition, latent_value in map(extract_condition_and_value, model_one_output[model_one_current_line:model_one_current_line+args.model_one_dimensions]):
                paradigm_file.write('{}\t{}\t{}\t{}\t{}'.format(
                    onset,
                    condition,
                    duration,
                    latent_value,
                    stimulus_filename
                ))
            for condition, latent_value in map(extract_condition_and_value, model_two_output[model_two_current_line:model_two_current_line + args.model_two_dimensions]):
                paradigm_file.write('{}\t{}\t{}\t{}\t{}'.format(
                    onset,
                    condition + args.model_one_dimensions,
                    duration,
                    latent_value,
                    stimulus_filename
                ))
            paradigm_file.write('{}\t{}\t{}\t{}\t{}'.format(
                onset,
                FACE_BIAS_CONDITION_ID,
                duration,
                DEFAULT_WEIGHT,
                stimulus_filename
            ))
            # Add one for the face bias condition
            model_one_current_line += args.model_one_dimensions + 1
            model_two_current_line += args.model_two_dimensions + 1
