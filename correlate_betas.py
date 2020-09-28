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

THRESHOLD = 3

subject_nums = range(1, 5)

MODEL_DIR = 'factor_vae_latent_24_gamma_24_try_1'
LATENT_VARIABLE_OP_NAME = 'sampled_latent_variable:0'
saved_model_dir = os.path.join('output', MODEL_DIR, 'tfhub')
sess = tf.Session()
model = tf.saved_model.load(export_dir=saved_model_dir, tags=[], sess=sess)


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

MODELS = ['dvae', 'vae']

for subject_num in subject_nums:
    print('Subject {}'.format(subject_num))
    localizer_map = sio.loadmat('localizer-maps/sub-{:02d}_sig.mat'.format(subject_num))['data'][0]
    n_voxels = len(localizer_map)

    for model in MODELS:
        print('Model {}'.format(model))
        betas = sio.loadmat('betas/sub-{:02d}_{}-beta.mat'.format(subject_num, model))
        betas = np.array(betas['data']).transpose()
        latent_betas = betas[:LATENT_DIMENSION]

        ground_truth_voxels = betas[LATENT_DIMENSION+1:].transpose()

        test_images = TEST_STIMULI['vaegan-sub-{:02d}-all'.format(subject_num)]

        predicted_voxels = np.empty((len(test_images), n_voxels))
        correlation = np.empty(n_voxels)
        for index, test_image in enumerate(test_images):
            celeba_file = os.path.join(args.celeba_dir, test_image)
            # (1 x 24)
            latent_values = encode_img(celeba_file).numpy()
            # (1 x n_voxels)
            predicted_voxels = np.matmul(latent_values, latent_betas)
            predicted_voxels[index] = predicted_voxels

        predicted_voxels = predicted_voxels.transpose()
        for i in range(len(correlation)):
            correlation[i] = spearmanr(predicted_voxels[i], ground_truth_voxels[i]).correlation

        below_threshold = localizer_map < THRESHOLD
        num_above_threshold = len(localizer_map) - np.sum(below_threshold)
        correlation[localizer_map < below_threshold] = 0
        average_correlation = np.sum(correlation) / num_above_threshold
        print('Correlation: {}'.format(average_correlation))