import numpy as np
import scipy.io as sio
import os
import argparse
from scipy.stats.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--functionals_dir', type=str, required=True)
parser.add_argument('--hemi', type=str, default='right', required=False, help='Which hemisphere to process. Must be one of [left, right, whole].')
args = parser.parse_args()

subject_nums = [1, 2, 3, 4]
ROIS = ['FFA', 'OFA', 'STS']

for roi in ROIS:
    print('ROI: {}'.format(roi))
    all_subject_predicted_voxels = []
    all_subject_ground_truth_voxels = []
    all_subject_average_correlations = []
    for subject_num in subject_nums:
        subject_dir = os.path.join(args.functionals_dir, 'vaegan-consolidated/unpackdata/vaegan-sub-{:02d}-all/bold/'.format(subject_num))
        correlations_dir = os.path.join(subject_dir, 'correlations')

        predicted_voxels = sio.loadmat(os.path.join(correlations_dir, '{}.{}.{}.predicted_voxels.mat'.format(args.model_name, roi, args.hemi)))
        predicted_voxels = predicted_voxels['data']
        all_subject_predicted_voxels.append(predicted_voxels)

        ground_truth_voxels = sio.loadmat(os.path.join(correlations_dir, '{}.{}.{}.ground_truth.mat'.format(args.model_name, roi, args.hemi)))
        ground_truth_voxels = ground_truth_voxels['data']
        all_subject_ground_truth_voxels.append(ground_truth_voxels)

        correlation = sio.loadmat(os.path.join(correlations_dir, '{}.{}.{}.correlations.mat'.format(args.model_name, roi, args.hemi)))
        correlation = correlation['data']
        all_subject_average_correlations.append(np.mean(correlation))

    average_correlation = np.mean(all_subject_average_correlations)
    print('Average correlation: {}'.format(average_correlation))

    all_subject_predicted_voxels = np.concatenate(all_subject_predicted_voxels)
    all_subject_ground_truth_voxels = np.concatenate(all_subject_ground_truth_voxels)

    assert all_subject_predicted_voxels.shape == all_subject_ground_truth_voxels.shape
    n_voxels = all_subject_predicted_voxels.shape[0]
    print('# voxels: {}'.format(n_voxels))

    print('generating null hypothesis')
    num_null_distribution_samples = 1000
    null_hypothesis_average_correlations = np.empty(num_null_distribution_samples)
    for null_distribution_sample_i in range(num_null_distribution_samples):
        if null_distribution_sample_i % 100 == 0:
            print('Generating {}/{}'.format(null_distribution_sample_i, num_null_distribution_samples))
        null_hypothesis_correlation = np.empty(n_voxels)
        null_hypothesis_ground_truth = all_subject_ground_truth_voxels.copy()
        # Shuffle the conditions within each voxels
        np.apply_along_axis(np.random.shuffle, 1, null_hypothesis_ground_truth)
        for voxel_i in range(n_voxels):
            null_hypothesis_correlation[voxel_i] = spearmanr(all_subject_predicted_voxels[voxel_i],
                                                       null_hypothesis_ground_truth[voxel_i]).correlation
        null_hypothesis_average_correlations[null_distribution_sample_i] = np.sum(null_hypothesis_correlation) / n_voxels
    print('Number of correlation>null hypothesis correlation')
    print(np.sum(average_correlation > null_hypothesis_average_correlations))
