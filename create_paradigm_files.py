import csv
import os
import re
import json

subjects_dir = 'subjects'
bold_dir = 'BOLD'

task_run_file_regex = 'task001_run.*.txt'
paradigm_file_name = 'dyn.para'

stimulus_mapping_file = 'stimuli/mapping.json'
json_file = open(stimulus_mapping_file, 'r')
stimulus_mapping = json.load(json_file)
filename_to_condition = stimulus_mapping['filename_to_condition_id']
null_condition = 0
null_stimulus_filename = 'i999.bmp'
circle_stimulus_filename = 'Circle.bmp'

ms_to_s = 1000.0

for root, subdirs, files in os.walk(subjects_dir):
    for file in files:
        # We found a matching file, open it and create the paradigm file
        if re.search(task_run_file_regex, file):
            with open(os.path.join(root, file), newline='') as csvfile:
                print('processing {}'.format(os.path.join(root, file)))
                initial_computer_time = None
                reader = csv.reader(csvfile)
                with open(os.path.join(root, paradigm_file_name), 'w') as paradigm_file:
                    for row in reader:
                        computer_time = int(row[1])
                        if not initial_computer_time:
                            initial_computer_time = computer_time
                        onset = (computer_time - initial_computer_time) / ms_to_s

                        stimulus_filename = os.path.basename(row[6])
                        if stimulus_filename == null_stimulus_filename \
                                or stimulus_filename == circle_stimulus_filename:
                            condition_id = null_condition
                        else:
                            condition_id = filename_to_condition[stimulus_filename]

                        stimulus_duration = int(row[4]) / ms_to_s
                        weight = 1.0

                        crosshair_duration = int(row[5]) / ms_to_s
                        paradigm_row = '{} {} {} {} {}\n'.format(
                            round(onset + crosshair_duration, 3),
                            condition_id,
                            stimulus_duration,
                            weight,
                            stimulus_filename
                        )
                        #print(paradigm_row)
                        paradigm_file.write(paradigm_row)
