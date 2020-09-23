import os
import argparse
import glob


FACE_RUNS_FILE_NAME = 'rlf_faces.txt'
SEEN_THRESHOLD = 3

parser = argparse.ArgumentParser()
parser.add_argument('--unpackdata_dir', type=str, help='', required=True)
args = parser.parse_args()

subject_nums = range(1, 5)

consolidated_subject_dir_template = 'vaegan-sub-{:02d}-all'

for subject_num in subject_nums:
    seen_images = {}
    consolidated_subject_name = consolidated_subject_dir_template.format(subject_num)
    consolidated_subject_dir = os.path.join(args.unpackdata_dir, consolidated_subject_name)
    consolidated_subject_bold_dir = os.path.join(consolidated_subject_dir, 'bold')

    # Create the localizer paradigm files
    for face_run in open(os.path.join(consolidated_subject_bold_dir, FACE_RUNS_FILE_NAME), 'r'):
        face_run_dir = os.path.join(consolidated_subject_bold_dir, '{:03d}'.format(int(face_run)))
        event_file = glob.glob(os.path.join(face_run_dir, '*.tsv'))[0]
        for event_line in open(event_file, 'r'):
            info = event_line.split()
            # Skip the header line
            if 'onset' in info:
                continue
            stimulus = info[3]
            # Get the file instead of the whole path
            stimulus = os.path.basename(stimulus)
            if stimulus not in seen_images:
                seen_images[stimulus] = 0
            seen_images[stimulus] += 1

    print('Subject {}'.format(subject_num))
    print(filter(lambda x: x[1] > SEEN_THRESHOLD, seen_images.items()))
