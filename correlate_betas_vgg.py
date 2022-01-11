import argparse
import os
import pickle

import scipy.io as sio
from scipy.stats.stats import spearmanr
import glob
import pathlib

import numpy as np
from keras_vggface.vggface import VGGFace
from keras.engine import Model
from keras_vggface import utils
from keras.preprocessing import image


parser = argparse.ArgumentParser()
parser.add_argument('--celeba_dir', type=str, help='', required=True)
parser.add_argument('--skip_p_value', action='store_true')
parser.add_argument('--subject_dir', type=str, required=True)
parser.add_argument('--functionals_dir', type=str, required=True)
parser.add_argument('--n_components', type=int, required=True)
parser.add_argument('--hidden_layer', type=str, required=True)
parser.add_argument('--tag', type=str, required=False)
parser.add_argument('--hemi', type=str, default='right', required=False, help='Which hemisphere to process. Must be one of [left, right, whole].')
args = parser.parse_args()

assert(args.hemi in ['left', 'right', 'whole'])

if args.tag:
    model_name = 'vgg.{}.{}.{}'.format(args.hidden_layer, args.n_components, args.tag)
else:
    model_name = 'vgg.{}.{}'.format(args.hidden_layer, args.n_components)

#THRESHOLDS = [2, 3, 4, 5]
subject_nums = [1, 2, 3, 4]

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

pca = pickle.load(open('{}.pkl'.format(model_name), 'rb'))
vgg_model = VGGFace()
out = vgg_model.get_layer(args.hidden_layer).output
vgg_model_new = Model(vgg_model.input, out)

def encode_img(img_file):
    img = image.load_img(img_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1)  # or version=2
    latent_encoding = vgg_model_new.predict(x)
    return pca.transform(latent_encoding)


stimulus_to_celeba = {}
for line in open('stimuli/ImageNames2Celeba.txt', 'r'):
    stimulus, celeba = line.split()
    stimulus_to_celeba[os.path.basename(stimulus)] = celeba

for subject_num in subject_nums:
    print('Subject {}'.format(subject_num))
    roi_dir = os.path.join(args.subject_dir, 'vaegan-sub-{:02d}-all'.format(subject_num), 'roi')

    left_roi_files = sorted(glob.glob(os.path.join(roi_dir, 'l*thresholded.both*.mat')))
    right_roi_files = sorted(glob.glob(os.path.join(roi_dir, 'r*thresholded.both*.mat')))

    test_images = TEST_STIMULI['vaegan-sub-{:02d}-all'.format(subject_num)]

    for left_roi_file, right_roi_file in zip(left_roi_files, right_roi_files):
        subject_dir = os.path.join(args.functionals_dir,
                                   'vaegan-consolidated/unpackdata/vaegan-sub-{:02d}-all/bold/'.format(subject_num))
        betas_location = os.path.join(subject_dir, '{}.betas.mat'.format(model_name))
        betas = sio.loadmat(betas_location)

        left_roi = os.path.basename(left_roi_file).split('.')[0]
        right_roi = os.path.basename(right_roi_file).split('.')[0]

        if args.hemi == 'left':
            localizer_map = sio.loadmat(left_roi_file)['threshold_roi'][0] > 0
            # Take the first n dimensions which correspond to voxels in the left hemi
            betas = np.array(betas['betas'][:len(localizer_map)][localizer_map]).transpose()
        elif args.hemi == 'right':
            localizer_map = sio.loadmat(right_roi_file)['threshold_roi'][0] > 0
            # Take the last n dimensions which correspond to voxels in the right hemi
            betas = np.array(betas['betas'][-len(localizer_map):][localizer_map]).transpose()
        else:
            left_localizer_map = sio.loadmat(left_roi_file)['threshold_roi'][0] > 0
            right_localizer_map = sio.loadmat(right_roi_file)['threshold_roi'][0] > 0
            localizer_map = np.concatenate((left_localizer_map, right_localizer_map))
            betas = np.array(betas['betas'][localizer_map]).transpose()

        n_voxels = np.sum(localizer_map)
        print('Number of voxels: {}'.format(n_voxels))

        latent_betas = betas[:args.n_components]
        bias_beta = betas[args.n_components]

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

        correlations_dir = os.path.join(subject_dir, 'correlations')
        pathlib.Path(correlations_dir).mkdir(parents=False, exist_ok=True)
        sio.savemat(os.path.join(correlations_dir, '{}.{}.{}.correlations.mat'.format(model_name, left_roi[1:], args.hemi)), {'data': correlation})

        average_correlation = np.mean(correlation)
        std = np.std(correlation)
        print('Correlation: {:.3f} \u00B1 {:.3f}'.format(average_correlation, std))

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
                # TODO save this array
            print('Number of correlation>null hypothesis correlation')
            print(np.sum(average_correlation > null_hypothesis_average_correlations))
