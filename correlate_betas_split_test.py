import argparse
import os

import numpy as np
import scipy.io as sio
from scipy.stats.stats import spearmanr
import tensorflow as tf
import glob
import pathlib

from disentanglement_lib.data.ground_truth.celeba import process_path

parser = argparse.ArgumentParser()
parser.add_argument('--celeba_dir', type=str, help='', required=True)
parser.add_argument('--skip_p_value', action='store_true')
parser.add_argument('--subject_dir', type=str, required=True)
parser.add_argument('--functionals_dir', type=str, required=True)

args = parser.parse_args()

#THRESHOLDS = [2, 3, 4, 5]
# TODO this variable shouldn't be used anymore
#model_type = 'dvae'
subject_nums = [1, 2, 3, 4]

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

stimulus_to_celeba = {}
for line in open('stimuli/ImageNames2Celeba.txt', 'r'):
    stimulus, celeba = line.split()
    stimulus_to_celeba[os.path.basename(stimulus)] = celeba

for subject_num in subject_nums:
    print('Subject {}'.format(subject_num))
    roi_dir = os.path.join(args.subject_dir, 'vaegan-sub-{:02d}-all'.format(subject_num), 'roi')

    left_roi_files = sorted(glob.glob(os.path.join(roi_dir, 'l*thresholded*.mat')))
    right_roi_files = sorted(glob.glob(os.path.join(roi_dir, 'r*thresholded*.mat')))

    test_images = TEST_STIMULI['vaegan-sub-{:02d}-all'.format(subject_num)]

    for left_roi_file, right_roi_file in zip(left_roi_files, right_roi_files):
        left_roi = os.path.basename(left_roi_file).split('.')[0]
        right_roi = os.path.basename(right_roi_file).split('.')[0]

        # Use > 0 to convert this to a boolean map
        left_localizer_map = sio.loadmat(left_roi_file)['threshold_roi'][0] > 0
        right_localizer_map = sio.loadmat(right_roi_file)['threshold_roi'][0] > 0
        whole_brain_localizer_map = np.concatenate((left_localizer_map, right_localizer_map))

        n_left_voxels = np.sum(left_localizer_map)
        n_right_voxels = np.sum(right_localizer_map)
        n_voxels = n_left_voxels + n_right_voxels
        print('Number of voxels in {}: {}'.format(left_roi, np.sum(left_localizer_map)))
        print('Number of voxels in {}: {}'.format(right_roi, np.sum(right_localizer_map)))

        subject_dir = os.path.join(args.functionals_dir, 'vaegan-consolidated/unpackdata/vaegan-sub-{:02d}-all/bold/'.format(subject_num))
        betas_location = os.path.join(subject_dir, 'vgg.fc7.24.split_test.betas.mat')
        betas = sio.loadmat(betas_location)

        betas = np.array(betas['betas'][whole_brain_localizer_map]).transpose()
        latent_betas = betas[:LATENT_DIMENSION]
        bias_beta = betas[LATENT_DIMENSION]

        ground_truth_voxels = betas[LATENT_DIMENSION+2:]
        print('Number of test conditions: {}'.format(len(ground_truth_voxels)))
        split_one_voxels = ground_truth_voxels[::2]
        split_two_voxels = ground_truth_voxels[1::2]

        split_one_voxels = split_one_voxels.transpose()
        split_two_voxels = split_two_voxels.transpose()

        correlation = np.empty(n_voxels)
        for i in range(len(correlation)):
            correlation[i] = spearmanr(split_one_voxels[i], split_two_voxels[i]).correlation

        correlations_dir = os.path.join(subject_dir, 'correlations')
        pathlib.Path(correlations_dir).mkdir(parents=False, exist_ok=True)
        sio.savemat(os.path.join(correlations_dir, '{}.{}.correlations.mat'.format('vgg.fc7.24.split_test', left_roi[1:])), {'data': correlation})
        left_correlation = np.sum(correlation[:n_left_voxels]) / n_left_voxels
        right_correlation = np.sum(correlation[n_left_voxels+1:]) / n_right_voxels
        average_correlation = np.sum(correlation) / n_voxels
        print('Correlation: {}'.format(average_correlation))
        print('Left correlation: {}'.format(left_correlation))
        print('Right correlation: {}'.format(right_correlation))

    # ThIS IS OLD CODE fROM SVhRM and does the whole brain
    '''
    localizer_map = sio.loadmat('localizer-maps/sub-{:02d}_sig.mat'.format(subject_num))['data'][0]
    for threshold in THRESHOLDS:
        print('Threshold {}'.format(threshold))
        above_threshold = localizer_map > threshold
        n_voxels = np.sum(above_threshold)
        print('Number above threshold: {}'.format(n_voxels))

        betas = sio.loadmat('betas/sub{:02d}_{}-beta.mat'.format(subject_num, model_type))
        betas = np.array(betas['data'][above_threshold]).transpose()
        latent_betas = betas[:LATENT_DIMENSION]
        bias_beta = betas[LATENT_DIMENSION]

        test_images = TEST_STIMULI['vaegan-sub-{:02d}-all'.format(subject_num)]

        # The final betas are for the test images
        ground_truth_voxels = betas[len(betas) - len(test_images):].transpose()

        predicted_voxels = np.empty((len(test_images), n_voxels))
        correlation = np.empty(n_voxels)
        for index, test_image in enumerate(test_images):
            celeba_file = os.path.join(args.celeba_dir, stimulus_to_celeba[test_image])
            # (1 x 24)
            latent_values = encode_img(celeba_file)
            # (1 x n_voxels)
            prediction = np.matmul(latent_values, latent_betas)
            predicted_voxels[index] = prediction + bias_beta

        predicted_voxels = predicted_voxels.transpose()
        for i in range(len(correlation)):
            correlation[i] = spearmanr(predicted_voxels[i], ground_truth_voxels[i]).correlation

        sio.savemat('subject{}-{}.mat'.format(subject_num, model_type), {'data': correlation})
        average_correlation = np.sum(correlation) / n_voxels
        print('Correlation: {}'.format(average_correlation))

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
                    null_hypothesis_correlation[j] = spearmanr(predicted_voxels[j], null_hypothesis_ground_truth[j]).correlation
                # Threshold the null hypothesis correlation
                null_hypothesis_average_correlations[i] = np.sum(null_hypothesis_correlation) / n_voxels
                # TODO save this array
            print('Number of correlation>null hypothesis correlation')
            print(np.sum(average_correlation > null_hypothesis_average_correlations))
            '''
