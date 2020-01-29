import os
import json

stimuli_dir = 'stimuli/mri'
files = sorted(os.listdir(stimuli_dir))

filename_to_condition_id = {}
condition_id_to_filename = {}

normal_face_filename_prefixes = ['f', 'pf', 'pu', 'u']
scrambled_face_filename_prefixes = ['ps', 's']

unused_stimuli = ['f076.bmp', 'f117.bmp', 'f147.bmp', 'f148.bmp', 'f149.bmp',
                  'f150.bmp', 'pf001.bmp', 'pf002.bmp', 'pf003.bmp', 'pf004.bmp',
                  'ps001.bmp', 'ps002.bmp', 'ps003.bmp', 'ps004.bmp', 'pu001.bmp',
                  'pu002.bmp', 'pu003.bmp', 'pu004.bmp', 's076.bmp', 's117.bmp',
                  's147.bmp', 's148.bmp', 's149.bmp', 's150.bmp', 'u076.bmp', 'u117.bmp']

normal_face_ids = []
scrambled_face_ids = []

index = 1
for file in files:
    if file in unused_stimuli:
        print('skipping {}'.format(file))
        continue
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

normal_face_parameters = ''
scrambled_face_parameters = ''
for id in normal_face_ids:
    normal_face_parameters += ' -a {}'.format(id)

for id in scrambled_face_ids:
    scrambled_face_parameters += ' -c {}'.format(id)
print('normal face ids:')
print(normal_face_parameters)
print('scrambled face ids:')
print(scrambled_face_parameters)

