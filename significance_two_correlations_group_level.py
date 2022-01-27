import argparse
import os

import numpy as np
import scipy.io as sio
from scipy.stats.stats import spearmanr
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--model1_name', type=str, required=True)
parser.add_argument('--model2_name', type=str, required=True)
parser.add_argument('--tag', type=str, required=False)
parser.add_argument('--functionals_dir', type=str, required=True)
parser.add_argument('--hemi', type=str, default='right', required=False, help='Which hemisphere to process. Must be one of [left, right, whole].')
args = parser.parse_args()

subject_nums = [1, 2, 3, 4]
ROIS = ['FFA', 'OFA', 'STS']


for roi in ROIS:
    print('ROI: {}'.format(roi))

    all_subject_model1_correlations = []
    all_subject_model2_correlations = []
    for subject_num in subject_nums:
        subject_dir = os.path.join(args.functionals_dir,
                                   'vaegan-consolidated/unpackdata/vaegan-sub-{:02d}-all/bold/'.format(subject_num))
        correlations_dir = os.path.join(subject_dir, 'correlations')

        model1_correlations = sio.loadmat(
            os.path.join(correlations_dir, '{}.{}.{}.correlations.mat'.format(args.model1_name, roi, args.hemi)))
        model1_correlations = model1_correlations['data'].squeeze()
        all_subject_model1_correlations.append(model1_correlations)

        model2_correlations = sio.loadmat(
            os.path.join(correlations_dir, '{}.{}.{}.correlations.mat'.format(args.model2_name, roi, args.hemi)))
        model2_correlations = model2_correlations['data'].squeeze()
        all_subject_model2_correlations.append(model2_correlations)

    average_model1_correlations = []
    average_model2_correlations = []
    for model1, model2 in zip(all_subject_model1_correlations, all_subject_model2_correlations):
        average_model1_correlations.append(np.mean(model1))
        average_model2_correlations.append(np.mean(model2))
    all_subject_model1_correlations = np.concatenate(all_subject_model1_correlations)
    all_subject_model2_correlations = np.concatenate(all_subject_model2_correlations)

    assert all_subject_model1_correlations.shape == all_subject_model2_correlations.shape

    num_null_distribution_samples = 1000

    randomized = np.random.randint(2, size=[num_null_distribution_samples, len(all_subject_model1_correlations)])
    test_vectors = all_subject_model1_correlations * randomized + all_subject_model2_correlations * (1-randomized)
    inverse_test_vectors = all_subject_model1_correlations * (1-randomized) + all_subject_model2_correlations * randomized

    test_vectors_difference = test_vectors - inverse_test_vectors
    null_hypothesis_average_correlations = np.average(test_vectors_difference, axis=1)

    average_model1_correlation = np.mean(average_model1_correlations)
    average_model2_correlation = np.mean(average_model2_correlations)
    average_difference = average_model1_correlation - average_model2_correlation

    print(np.sum(average_difference > null_hypothesis_average_correlations))

'''
For every voxel in ROI (ex FFA has 100 voxels)
FactorVAE we have 100 correlation values
VGG we have 100 correlation values

Are the Factor values higher than the VGG values?
Generate a test vector where for each of the 100 voxels, we randomly choose the value from the correpsonding voxel in either Factor or VGG
ex: 
test_vector1 = [factor1, factor2, factor3, vgg4, factor5, vgg6,....factor100]
Generate the opposite vector
test_vector2 = [vgg1, vgg2, vgg3, factor4, vgg5, factor6,...vgg100]

test_vector = test_vector1-test_vector2
avg(test_vector)

Do this 1000 times to generate our null distribution

Then take
difference = factor-vgg (1x100)
avg(difference)
print(np.sum(difference > null_dist))
'''