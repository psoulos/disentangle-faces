"""
This file calculates the voxel correlations for combined models where the latent dimensions from two models are
concatenated together.
"""

import argparse
import os

import numpy as np
import scipy.io as sio
from scipy.stats.stats import spearmanr
import glob
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--celeba_dir', type=str, help='', required=True)
parser.add_argument('--skip_p_value', action='store_true')
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--subject_dir', type=str, required=True)
parser.add_argument('--functionals_dir', type=str, required=True)
parser.add_argument('--hemi', type=str, default='right', required=False, help='Which hemisphere to process. Must be one of [left, right, whole].')
parser.add_argument('--skip_localizer', action='store_true', help='DEPRECATED. Use `--localizer all` instead.')
parser.add_argument('--localizer', type=str, required=False, default='roi', help='One of [roi,score,all].')
args = parser.parse_args()

assert(args.hemi in ['left', 'right', 'whole'])
assert(args.localizer in ['roi', 'score', 'all'])

subject_nums = [1, 2, 3, 4]

LOCALIZER_TYPE_TO_KEY_NAME = {
    'roi': 'threshold_roi',
    'score': '{}_score',
    'all': '{}_score'
}

precomputed_latent_values = {}

def encode_img(img_file):
    latent_value = []
    if contains_factor:
        latent_value.append(precomputed_latent_values['factor'][img_file])
    if contains_vae:
        latent_value.append(precomputed_latent_values['vae'][img_file])
    if contains_vgg:
        latent_value.append(precomputed_latent_values['vgg'][img_file])

    return np.concatenate(latent_value, axis=1)


LATENT_DIMENSION = 24
TEST_STIMULI = {
    'vaegan-sub-01-all': ['M2553.jpg', 'F1631.jpg', 'M2424.jpg', 'F1235.jpg', 'F1148.jpg', 'M2156.jpg', 'F2376.jpg',
                          'M1584.jpg', 'M2466.jpg', 'F2068.jpg', 'F1586.jpg', 'F1232.jpg', 'M2203.jpg', 'M1365.jpg',
                          'M2248.jpg', 'F2467.jpg', 'M2336.jpg', 'F1145.jpg', 'F2377.jpg', 'M2246.jpg'],
    'vaegan-sub-02-all': ['M7260.jpg', 'M8712.jpg', 'M7704.jpg', 'F7216.jpg', 'F8408.jpg', 'F6117.jpg', 'F7792.jpg',
                          'F8669.jpg', 'M4446.jpg', 'M4535.jpg', 'M6338.jpg', 'M7041.jpg', 'F5414.jpg', 'M6776.jpg',
                          'M4621.jpg', 'F4622.jpg', 'F6118.jpg', 'M6953.jpg', 'F5724.jpg', 'F6116.jpg'],
    'vaegan-sub-03-all': ['M10035.jpg', 'F09021.jpg', 'M12366.jpg', 'M10124.jpg', 'M11003.jpg', 'F09903.jpg',
                          'F12323.jpg', 'F10912.jpg', 'M10165.jpg', 'F09152.jpg', 'F11266.jpg', 'M10783.jpg',
                          'F09109.jpg', 'M12233.jpg', 'F11440.jpg', 'M13160.jpg', 'F11400.jpg', 'F08933.jpg',
                          'M08800.jpg', 'M11927.jpg'],
    'vaegan-sub-04-all': ['F14697.jpg', 'F14081.jpg', 'M14039.jpg', 'F13996.jpg', 'M17160.jpg', 'F15049.jpg',
                          'F15137.jpg', 'M13644.jpg', 'M15665.jpg', 'M13289.jpg', 'M15488.jpg', 'M17204.jpg',
                          'F15976.jpg', 'M16368.jpg', 'F13774.jpg', 'F15404.jpg', 'M17336.jpg', 'F13640.jpg',
                          'F14520.jpg', 'M14256.jpg']
}

model_name = args.model_name

contains_factor = '.factor_vae.' in model_name
contains_vgg = '.vgg.' in model_name
contains_vae = '.vae.' in model_name

stimulus_to_celeba = {}
for line in open('../stimuli/ImageNames2Celeba.txt', 'r'):
    stimulus, celeba = line.split()
    stimulus_to_celeba[os.path.basename(stimulus)] = celeba

for subject_num in subject_nums:
    print('Subject {}'.format(subject_num))
    roi_dir = os.path.join(args.subject_dir, 'vaegan-sub-{:02d}-all'.format(subject_num), 'roi')

    if args.localizer == 'roi':
        left_localizer_filename = 'l*thresholded.both*.mat'
        right_localizer_filename = 'r*thresholded.both*.mat'
    elif args.localizer == 'score':
        left_localizer_filename = 'whole_brain_score_1.5.lh.surf.thresholded.mat'
        right_localizer_filename = 'whole_brain_score_1.5.rh.surf.thresholded.mat'
        localizer_name = 'score'
    else:
        # Use one of the other filenames to figure out how many voxels are in each hemisphere but later we will set the
        # localizer to all ones
        left_localizer_filename = 'whole_brain_score_1.5.lh.surf.thresholded.mat'
        right_localizer_filename = 'whole_brain_score_1.5.rh.surf.thresholded.mat'
        localizer_name = 'all'

    left_localizer_files = sorted(glob.glob(os.path.join(roi_dir, left_localizer_filename)))
    right_localizer_files = sorted(glob.glob(os.path.join(roi_dir, right_localizer_filename)))

    test_images = TEST_STIMULI['vaegan-sub-{:02d}-all'.format(subject_num)]

    for left_localizer_file, right_localizer_file in zip(left_localizer_files, right_localizer_files):
        subject_dir = os.path.join(args.functionals_dir, 'vaegan-consolidated/unpackdata/vaegan-sub-{:02d}-all/bold/'.format(subject_num))
        betas_location = os.path.join(subject_dir, '{}.betas.mat'.format(model_name))
        latent_values_dir = os.path.join(subject_dir, 'correlations')
        precomputed_latent_values['factor'] = sio.loadmat(os.path.join(latent_values_dir, 'factor_vae_output.mat'))
        precomputed_latent_values['vae'] = sio.loadmat(os.path.join(latent_values_dir, 'vae_output.mat'))
        precomputed_latent_values['vgg'] = sio.loadmat(os.path.join(latent_values_dir, 'vgg_output.mat'))

        betas = sio.loadmat(betas_location)

        if args.localizer == 'roi':
            # Use the roi file name to extract the roi name
            localizer_name = os.path.basename(left_localizer_file).split('.')[0][1:]
            print(localizer_name)

        if args.hemi == 'left':
            localizer_map = sio.loadmat(left_localizer_file)
            localizer_map = localizer_map[LOCALIZER_TYPE_TO_KEY_NAME[args.localizer].format(args.hemi)][0] > 0
            # Use all voxels if we don't want to use the localizer
            if args.localizer == 'all':
                localizer_map = np.ones_like(localizer_map)
            # Take the first n dimensions which correspond to voxels in the left hemi
            betas = np.array(betas['betas'][:len(localizer_map)][localizer_map]).transpose()
        elif args.hemi == 'right':
            localizer_map = sio.loadmat(right_localizer_file)
            localizer_map = localizer_map[LOCALIZER_TYPE_TO_KEY_NAME[args.localizer].format(args.hemi)][0] > 0
            # Use all voxels if we don't want to use the localizer
            if args.localizer == 'all':
                localizer_map = np.ones_like(localizer_map)
            # Take the last n dimensions which correspond to voxels in the right hemi
            betas = np.array(betas['betas'][-len(localizer_map):][localizer_map]).transpose()
        else:
            left_localizer_map = sio.loadmat(left_localizer_file)
            left_localizer_map = left_localizer_map[LOCALIZER_TYPE_TO_KEY_NAME[args.localizer].format('left')][0] > 0
            n_left_voxels = np.sum(left_localizer_map)
            right_localizer_map = sio.loadmat(right_localizer_file)
            right_localizer_map = right_localizer_map[LOCALIZER_TYPE_TO_KEY_NAME[args.localizer].format('right')][0] > 0
            localizer_map = np.concatenate((left_localizer_map, right_localizer_map))
            # Use all voxels if we don't want to use the localizer
            if args.localizer == 'all':
                localizer_map = np.ones_like(localizer_map)
            betas = np.array(betas['betas'][localizer_map]).transpose()

        n_voxels = np.sum(localizer_map)
        print('Number of voxels: {}'.format(n_voxels))

        latent_betas = betas[:LATENT_DIMENSION]
        bias_beta = betas[LATENT_DIMENSION]

        # The final betas are for the test images
        ground_truth_voxels = betas[len(betas) - len(test_images):].transpose()

        predicted_voxels = np.empty((len(test_images), n_voxels))
        correlation = np.empty(n_voxels)
        #test_image_to_latent_values = {}
        for index, test_image in enumerate(test_images):
            # (1 x 24)
            latent_values = encode_img(test_image.split('.jpg')[0])
            #test_image_to_latent_values[test_image.split('.')[0]] = latent_values
            # (1 x n_voxels)
            prediction = np.matmul(latent_values, latent_betas)
            predicted_voxels[index] = prediction + bias_beta

        predicted_voxels = predicted_voxels.transpose()
        for i in range(len(correlation)):
            correlation[i] = spearmanr(predicted_voxels[i], ground_truth_voxels[i]).correlation
        correlation[np.isnan(correlation)] = 0

        correlations_dir = os.path.join(subject_dir, 'correlations')
        pathlib.Path(correlations_dir).mkdir(parents=False, exist_ok=True)
        #sio.savemat(os.path.join(correlations_dir, '{}.{}.{}.test_image_latent_values.mat'.format(model_name, localizer_name, args.hemi)), test_image_to_latent_values)
        sio.savemat(os.path.join(correlations_dir, '{}.{}.{}.predicted_voxels.mat'.format(model_name, localizer_name, args.hemi)), {'data': predicted_voxels})
        sio.savemat(os.path.join(correlations_dir, '{}.{}.{}.ground_truth.mat'.format(model_name, localizer_name, args.hemi)), {'data': ground_truth_voxels})
        sio.savemat(os.path.join(correlations_dir, '{}.{}.{}.correlations.mat'.format(model_name, localizer_name, args.hemi)), {'data': correlation})

        average_correlation = np.mean(correlation)
        std = np.std(correlation)
        print('Correlation: {:.3f} \u00B1 {:.3f}'.format(average_correlation, std))
        if args.hemi == 'whole':
            left_correlation = correlation[:n_left_voxels]
            right_correlation = correlation[n_left_voxels:]
            print('Left correlation: {:.3f} \u00B1 {:.3f}'.format(np.mean(left_correlation), np.std(left_correlation)))
            print('Right correlation: {:.3f} \u00B1 {:.3f}'.format(np.mean(right_correlation), np.std(right_correlation)))

        if not args.skip_p_value:
            # Generate samples from a null distribution for significance testing
            print('generating null hypothesis')
            num_null_distribution_samples = 1000
            null_hypothesis_average_correlations = np.empty(num_null_distribution_samples)
            for i in range(num_null_distribution_samples):
                null_hypothesis_correlation = np.empty(n_voxels)
                null_hypothesis_ground_truth = ground_truth_voxels.copy()
                # Shuffle the conditions within each voxels
                np.apply_along_axis(np.random.shuffle, 1, null_hypothesis_ground_truth)
                for j in range(len(correlation)):
                    null_hypothesis_correlation[j] = spearmanr(predicted_voxels[j],
                                                               null_hypothesis_ground_truth[j]).correlation
                # Threshold the null hypothesis correlation
                null_hypothesis_average_correlations[i] = np.sum(null_hypothesis_correlation) / n_voxels
            print('Number of correlation>null hypothesis correlation')
            print(np.sum(average_correlation > null_hypothesis_average_correlations))
