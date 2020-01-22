import os
import json

stimuli_dir = 'stimuli/mri'
files = sorted(os.listdir(stimuli_dir))

filename_to_condition_id = {}
condition_id_to_filename = {}

normal_face_filename_prefixes = ['f', 'pf', 'pu', 'u']
scrambled_face_filename_prefixes = ['ps', 's']

normal_face_ids = []
scrambled_face_ids = []

index = 1
for file in files:
    filename_to_condition_id[file] = index
    condition_id_to_filename[index] = file
    for prefix in normal_face_filename_prefixes:
        if file.startswith(prefix):
            normal_face_ids.append(index)

    for prefix in scrambled_face_filename_prefixes:
        if file.startswith(prefix):
            scrambled_face_ids.append(index)

    index += 1

results = {}
results['filename_to_condition_id'] = filename_to_condition_id
results['condition_id_to_filename'] = condition_id_to_filename
results['normal_face_ids'] = normal_face_ids
results['scrambled_face_ids'] = scrambled_face_ids

with open('stimuli/mapping.json', 'w') as f:
    f.write(json.dumps(results, indent=4))


