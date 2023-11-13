import argparse
import os
import glob
import shutil
# Create vaegan-sub-XX-all
#          - bold
#          - subjectname
# Iterate through all vaegan-sub-XX-ses-Y/bold
# Keep an incrementing tally of the current run ID and copy the runs into all/bold with this
#      run ID
# Open vaegan-sub-XX-ses-Y/bold/rlf_faces.txt and write the new run ID to vaegan-sub-XX-all/bold/rlf_faces.txt
# Do the same with rlf_localizer.txt

parser = argparse.ArgumentParser()
parser.add_argument('--unpackdata_dir', type=str, help='', required=True)
args = parser.parse_args()

subject_nums = range(1, 5)
session_nums = range(1, 9)

subject_session_dir_template = 'vaegan-sub-{:02d}-ses-{:02d}'
consolidated_subject_dir_template = 'vaegan-sub-{:02d}-all'

for subject_num in subject_nums:
    consolidated_subject_name = consolidated_subject_dir_template.format(subject_num)
    consolidated_subject_dir = os.path.join(args.unpackdata_dir, consolidated_subject_name)
    os.makedirs(consolidated_subject_dir)
    with open(os.path.join(consolidated_subject_dir, 'subjectname'), 'w') as f:
        f.write(consolidated_subject_name)
    consolidated_subject_bold_dir = os.path.join(consolidated_subject_dir, 'bold')
    os.makedirs(consolidated_subject_bold_dir)

    consolidated_localizer_run_file = open(os.path.join(consolidated_subject_bold_dir, 'rlf_localizer.txt'), 'w')
    consolidated_face_run_file = open(os.path.join(consolidated_subject_bold_dir, 'rlf_faces.txt'), 'w')

    consolidated_run_num = 1
    for session_num in session_nums:
        subject_session_dir = subject_session_dir_template.format(subject_num, session_num)
        subject_session_bold_dir = os.path.join(args.unpackdata_dir, subject_session_dir, 'bold')
        localizer_runs = []
        face_runs = []

        for line in open(os.path.join(subject_session_bold_dir, 'rlf_localizer.txt'), 'r'):
            localizer_runs.append(line.strip())
        for line in open(os.path.join(subject_session_bold_dir, 'rlf_faces.txt'), 'r'):
            face_runs.append(line.strip())

        functional_runs = filter(lambda x: x.isdigit(), sorted(os.listdir(subject_session_bold_dir)))
        for functional_run in functional_runs:
            if functional_run in localizer_runs:
                consolidated_localizer_run_file.write('{:03d}\n'.format(consolidated_run_num))
            elif functional_run in face_runs:
                consolidated_face_run_file.write('{:03d}\n'.format(consolidated_run_num))
            else:
                raise NameError('functional run {} is neither localizer or face'.format(functional_run))

            functional_run_dir = os.path.join(subject_session_bold_dir, functional_run)

            consolidated_functional_run = os.path.join(consolidated_subject_bold_dir,
                                                       '{:03d}'.format(consolidated_run_num))
            os.makedirs(consolidated_functional_run)
            consolidated_run_num += 1

            shutil.copy(os.path.join(functional_run_dir, 'f.nii.gz'), consolidated_functional_run)
            shutil.copy(glob.glob(os.path.join(functional_run_dir, '*.tsv'))[0], consolidated_functional_run)
