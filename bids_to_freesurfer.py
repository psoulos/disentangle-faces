import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--bids_dir', type=str, help='The location of the BIDS directory')
parser.add_argument('--output_dir', type=str, help='The location of the output directory')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

for subject_dir in os.listdir(args.bids_dir):
    print('Processing {}'.format(subject_dir))
    subject_path = os.path.join(args.bids_dir, subject_dir)
    for sess_dir in os.listdir(subject_path):
        print('Processing session {}'.format(sess_dir))
        freesurfer_subject_name = 'vaegan-{}-{}'.format(subject_dir, sess_dir)
        freesurfer_subject_dir = os.path.join(
            args.output_dir, freesurfer_subject_name)
        os.makedirs(freesurfer_subject_dir)

        # Copy the anatomical files
        freesurfer_anatomical_dir = os.path.join(freesurfer_subject_dir, 'anatomy')
        bids_anatomical_dir = os.path.join(subject_path, sess_dir, 'anat')
        os.makedirs(freesurfer_anatomical_dir)
        for f in os.listdir(bids_anatomical_dir):
            shutil.copy(os.path.join(bids_anatomical_dir, f), freesurfer_anatomical_dir)

        # Copy and rename the bold files
        freesurfer_bold_dir = os.path.join(freesurfer_subject_dir, 'bold')
        bids_bold_dir = os.path.join(subject_path, sess_dir, 'func')
        os.makedirs(freesurfer_bold_dir)
        # Print the original BIDS runs into a file for inspecting later
        with open(os.path.join(freesurfer_bold_dir, 'orig_run_names.txt'), 'w') as f:
            for i in sorted(os.listdir(bids_bold_dir)):
                f.write('{}\n'.format(i))

        # Increment this for each run
        run_num = 1



        # Create the subjectname file
        with open(os.path.join(freesurfer_subject_dir, 'subjectname'), 'w') as f:
            f.write(freesurfer_subject_name)
