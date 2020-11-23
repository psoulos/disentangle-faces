import argparse
import os

import numpy as np
import scipy.io as sio
from scipy.stats.stats import spearmanr
import tensorflow as tf

from disentanglement_lib.data.ground_truth.celeba import process_path

parser = argparse.ArgumentParser()
parser.add_argument('--celeba_dir', type=str, help='', required=True)
args = parser.parse_args()

THRESHOLDS = [2, 3, 4]

subject_nums = [1, 3, 4]

MODELS = {
    'dvae': 'factor_vae_latent_24_gamma_24_try_1',
    'vae': 'vae_latent_24_try_1'
}
LATENT_VARIABLE_OP_NAME = 'encoder/means/BiasAdd:0'
sess = tf.Session()


def encode_img(img_file):
    input_image = process_path(img_file, size=(64, 64))
    # We need to add an extra dimension for a batch size of 1
    input_image = np.expand_dims(input_image, axis=0)
    latent_variable = sess.run(LATENT_VARIABLE_OP_NAME, feed_dict={'Placeholder:0': input_image})
    # The first index is the batch of size 1 so index into that
    return latent_variable[0]


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

model_type = 'dvae'
print('Model {}'.format(model_type))
saved_model_dir = os.path.join('output', MODELS[model_type], 'tfhub')
model = tf.saved_model.load(export_dir=saved_model_dir, tags=[], sess=sess)

stimulus_to_celeba = {}
for line in open('stimuli/ImageNames2Celeba.txt', 'r'):
    stimulus, celeba = line.split()
    stimulus_to_celeba[os.path.basename(stimulus)] = celeba

#correlation_vectors = {}
for subject_num in subject_nums:
    print('Subject {}'.format(subject_num))
    localizer_map = sio.loadmat('localizer-maps/sub-{:02d}_sig.mat'.format(subject_num))['data'][0]
    threshold = 4
    above_threshold = localizer_map > threshold
    n_voxels = np.sum(above_threshold)
    print('Number above threshold: {}'.format(n_voxels))

    print('Model {}'.format(model_type))
    betas = sio.loadmat('betas/sub-{:02d}_mean-{}-beta.mat'.format(subject_num, model_type))
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

    '''
    filtered_correlation = correlation.copy()
    filtered_correlation[below_threshold] = 0
    sio.savemat('threshold-{}-subject{}-{}.mat'.format(threshold, subject_num, model_type),
                {'data': filtered_correlation})
    average_correlation = np.sum(filtered_correlation) / num_above_threshold
    print('Correlation: {}'.format(average_correlation))
    '''

    # Generate samples from a null distribution for significance testing
    print('generated null hypothesis')
    num_null_distribution_samples = 1000
    null_hypothesis_average_correlations = np.empty(num_null_distribution_samples)
    null_samples = np.empty((num_null_distribution_samples, len(test_images), n_voxels))
    for i in range(num_null_distribution_samples):
        null_hypothesis_correlation = np.empty(n_voxels)
        null_hypothesis_ground_truth = ground_truth_voxels.copy()
        np.random.shuffle(null_hypothesis_ground_truth)
        for j in range(len(correlation)):
            null_hypothesis_correlation[j] = spearmanr(predicted_voxels[j], null_hypothesis_ground_truth[j]).correlation
        # Threshold the null hypothesis corerlation
        null_hypothesis_average_correlations[i] = np.sum(null_hypothesis_correlation) / n_voxels

    #correlation_vectors['subject{}-{}'.format(subject_num, model_type)] = correlation.copy()

    sio.savemat('subject{}-{}.mat'.format(subject_num, model_type), {'data': correlation})
    average_correlation = np.sum(correlation) / n_voxels
    print('Correlation: {}'.format(average_correlation))

    print('Number of correlation>null hypothesis correlation')
    print(np.sum(average_correlation > null_hypothesis_average_correlations))
    '''
    for threshold in THRESHOLDS:
        print('Threshold {}'.format(threshold))
        filtered_correlation = correlation.copy()
        below_threshold = localizer_map < threshold
        num_above_threshold = len(localizer_map) - np.sum(below_threshold)
        print('Number above threshold: {}'.format(num_above_threshold))
        filtered_correlation[below_threshold] = 0
        sio.savemat('threshold-{}-subject{}-{}.mat'.format(threshold, subject_num, model_type), {'data': filtered_correlation})
        #correlation_vectors['filtered-subject{}-{}'.format(subject_num, model_type)] = filtered_correlation
        average_correlation = np.sum(filtered_correlation) / num_above_threshold
        print('Correlation: {}'.format(average_correlation))
    '''
#sio.savemat('correlation.mat', correlation_vectors)

