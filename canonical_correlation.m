subject_nums = {1 2 3 4};
NUM_TEST_IMAGES_PER_SUBEJCT = 20;
NUM_TEST_IMAGES = NUM_TEST_IMAGES_PER_SUBEJCT * size(subject_nums,2);
NUM_DIMENSIONS = 24;
vae = zeros(NUM_TEST_IMAGES,NUM_DIMENSIONS);
factor = zeros(NUM_TEST_IMAGES,NUM_DIMENSIONS);
vgg = zeros(NUM_TEST_IMAGES,NUM_DIMENSIONS);
for i = 1:length(subject_nums)
    subject_num = subject_nums{i};
    vae((subject_num-1)*NUM_TEST_IMAGES_PER_SUBEJCT+1:subject_num*NUM_TEST_IMAGES_PER_SUBEJCT,:) = cell2mat(struct2cell(load(['/Users/psoulos/bin/freesurfer/sessions/vaegan-consolidated/unpackdata/vaegan-sub-0' num2str(subject_num) '-all/bold/correlations/vae_output.mat'])));
    factor((subject_num-1)*NUM_TEST_IMAGES_PER_SUBEJCT+1:subject_num*NUM_TEST_IMAGES_PER_SUBEJCT,:) = cell2mat(struct2cell(load(['/Users/psoulos/bin/freesurfer/sessions/vaegan-consolidated/unpackdata/vaegan-sub-0' num2str(subject_num) '-all/bold/correlations/factor_vae_output.mat'])));
    vgg((subject_num-1)*NUM_TEST_IMAGES_PER_SUBEJCT+1:subject_num*NUM_TEST_IMAGES_PER_SUBEJCT,:) = cell2mat(struct2cell(load(['/Users/psoulos/bin/freesurfer/sessions/vaegan-consolidated/unpackdata/vaegan-sub-0' num2str(subject_num) '-all/bold/correlations/vgg_output.mat'])));
end
[A,B,r,U,V] = canoncorr(vae, factor);
['CCA: Factor and VAE: ' num2str(mean(r))]

[A,B,r,U,V] = canoncorr(vae, vgg);
['CCA: VAE and VGG: ' num2str(mean(r))]

[A,B,r,U,V] = canoncorr(factor, vgg);
['CCA: Factor and VGG: ' num2str(mean(r))]