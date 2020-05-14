import os
import argparse
import shutil
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--bids_dir', type=str, help='The location of the BIDS directory')
parser.add_argument('--freesurfer_dir', type=str, help='The location of the output freesurfer '
                                                       'directory')
args = parser.parse_args()

if not os.path.exists(args.freesurfer_dir):
    os.makedirs(args.freesurfer_dir)

for subject_dir in os.listdir(args.bids_dir):
    print('Processing {}'.format(subject_dir))
    subject_path = os.path.join(args.bids_dir, subject_dir)
    for sess_dir in os.listdir(subject_path):
        print('Processing session {}'.format(sess_dir))
        freesurfer_subject_name = 'vaegan-{}-{}'.format(subject_dir, sess_dir)
        freesurfer_subject_dir = os.path.join(
            args.freesurfer_dir, freesurfer_subject_name)
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

        localizer_runs = sorted(glob.glob(os.path.join(bids_bold_dir, '*loc*.nii.gz')))
        functional_runs = sorted(glob.glob(os.path.join(bids_bold_dir, '*faces*.nii.gz')))

        orig_run_names_file = open(os.path.join(freesurfer_bold_dir, 'orig_run_names.txt'), 'w')
        localizer_run_list_file = open(os.path.join(freesurfer_bold_dir, 'rlf_localizer.txt'), 'w')
        faces_run_list_file = open(os.path.join(freesurfer_bold_dir, 'rlf_faces.txt'), 'w')
        # Increment this for each run
        run_num = 1

        # Process the localizer runs
        for localizer_run in localizer_runs:
            orig_run_names_file.write('{:03d}: {}\n'.format(run_num, localizer_run))
            localizer_run_list_file.write('{:03d}\n'.format(run_num))
            freesurfer_localizer_dir = os.path.join(freesurfer_bold_dir, '{:03d}'.format(run_num))
            os.makedirs(freesurfer_localizer_dir)
            shutil.copyfile(localizer_run, os.path.join(freesurfer_localizer_dir, 'f.nii.gz'))
            shutil.copy(localizer_run.replace('.nii.gz', '.json'), freesurfer_localizer_dir)
            shutil.copy(localizer_run.replace('_bold.nii.gz', '_events.tsv'), freesurfer_localizer_dir)
            run_num += 1

        # Process the functional runs
        for functional_run in functional_runs:
            orig_run_names_file.write('{:03d}: {}\n'.format(run_num, functional_run))
            faces_run_list_file.write('{:03d}\n'.format(run_num))
            freesurfer_functional_dir = os.path.join(freesurfer_bold_dir, '{:03d}'.format(run_num))
            os.makedirs(freesurfer_functional_dir)
            shutil.copyfile(functional_run, os.path.join(freesurfer_functional_dir, 'f.nii.gz'))
            shutil.copy(functional_run.replace('.nii.gz', '.json'), freesurfer_functional_dir)
            shutil.copy(functional_run.replace('_bold.nii.gz', '_events.tsv'),
                        freesurfer_functional_dir)
            run_num += 1

        orig_run_names_file.close()
        faces_run_list_file.close()
        localizer_run_list_file.close()

        # Create the subjectname file
        with open(os.path.join(freesurfer_subject_dir, 'subjectname'), 'w') as f:
            f.write(freesurfer_subject_name)
