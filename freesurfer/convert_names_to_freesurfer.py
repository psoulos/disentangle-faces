import os
import re

'''
The data repository does not use the expected filenames for freesurfer.
This script iterates through the data and changes the filenames to the expected names.

1. Each $SUBJECT/BOLD directory is renamed to $SUBJECT/bold
2. Each $SUBJECT/bold/task001_run00* is renamed to 00*
3. Create $SUBJECT/bold/rlf.txt which is the run file containing all of the runs 00*
4. Each $SUBECT/bold/00*/bold.nii.gz is renamed to f.nii.gz

We walk over the directory hierarchy a few times which may not be optimal, but I'm not sure
how walking works if we are changing the names of files and directories during the walk. Each
walk makes one of the four changes listed above.
'''

subjects_dir = 'subjects'

bold_dir_original = 'BOLD'
bold_dir_rename = 'bold'

task_dir_original_regex = 'task001_run.*.'
#task_dir_rename = '{0:0=3d}'

run_list_filename = 'rlf.txt'

bold_response_original = 'bold.nii.gz'
bold_response_rename = 'f.nii.gz'

for root, subdirs, files in os.walk(subjects_dir):
    for subdir in subdirs:
        if subdir == bold_dir_original:
            print('renaming {} to {}'.format(os.path.join(root, subdir),
                                             os.path.join(root, bold_dir_rename)))
            os.rename(os.path.join(root, bold_dir_original), os.path.join(root, bold_dir_rename))

for root, subdirs, files in os.walk(subjects_dir):
    for subdir in subdirs:
        if re.search(task_dir_original_regex, subdir):
            rename = subdir[-3:]
            print('renaming {} to {}'.format(os.path.join(root, subdir), os.path.join(root, rename)))
            os.rename(os.path.join(root, subdir), os.path.join(root, rename))

for root, subdirs, files in os.walk(subjects_dir):
    for subdir in subdirs:
        if subdir == bold_dir_rename:
            with open(os.path.join(
                    os.path.join(root, subdir), run_list_filename), 'w') as run_list_file:
                for i in range(1, 10):
                    run_list_file.write('{0:0=3d}\n'.format(i))

for root, subdirs, files in os.walk(subjects_dir):
    for file in files:
        if file == bold_response_original:
            print('renaming {} to {}'.format(os.path.join(root, file),
                                             os.path.join(root, bold_response_rename)))
            os.rename(os.path.join(root, file), os.path.join(root, bold_response_rename))