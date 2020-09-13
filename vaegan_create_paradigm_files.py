import os
import argparse
import glob


LOCALIZER_RUNS_FILE_NAME = 'rlf_localizer.txt'
FACE_RUNS_FILE_NAME = 'rlf_faces.txt'
CONDITION_IDS = {
    'fix': 0,
    'face': 1,
    'object': 2
}

parser = argparse.ArgumentParser()
parser.add_argument('--unpackdata_dir', type=str, help='', required=True)
args = parser.parse_args()

subject_nums = range(1, 5)

consolidated_subject_dir_template = 'vaegan-sub-{:02d}-all'

for subject_num in subject_nums:
    consolidated_subject_name = consolidated_subject_dir_template.format(subject_num)
    consolidated_subject_dir = os.path.join(args.unpackdata_dir, consolidated_subject_name)
    consolidated_subject_bold_dir = os.path.join(consolidated_subject_dir, 'bold')

    # Create the localizer paradigm files
    for localizer_run in os.path.join(consolidated_subject_bold_dir, LOCALIZER_RUNS_FILE_NAME):
        localizer_run_dir = os.path.join(consolidated_subject_bold_dir, '{:03d}'.format(localizer_run))
        # Convert the events file from OpenNeuro to Freesurfer paradigm format
        paradigm_file = open(os.path.join(localizer_run_dir, 'dyn.para'), 'w')
        event_file = glob.glob('*.tsv')[0]
        onset = 0
        for event_line in open(event_file, 'r'):
            info = event_line.split()
            # Skip the header line
            if 'onset' in info:
                continue
            duration = info[1]
            condition_type = info[2]
            condition_id = CONDITION_IDS[condition_type]
            paradigm_file.write('{}\t{}\t{}\t{}\n'.format(
                onset,
                condition_id,
                duration,
                condition_type
            ))
            onset += duration

    # TODO Create the face run paradigm files