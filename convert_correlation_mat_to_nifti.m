assert(~(getenv('FIELDTRIP_DIR') == ""), 'You must first set the environment variable FIELDTRIP_DIR')
assert(~(getenv('FUNCTIONALS_DIR') == ""), 'You must first set the environment variable FUNCTIONALS_DIR')
assert(~(getenv('SUBJECTS_DIR') == ""), 'You must first set the environment variable SUBJECTS_DIR')
assert(~(getenv('MODEL_NAME') == ""), 'You must first set the environment variable MODEL_NAME')

addpath([getenv('FIELDTRIP_DIR') '/external/freesurfer'])
subject_nums = {1 2 3 4};
hemis = {'l' 'r'};
rois = {'FFA' 'OFA' 'STS'};
for i = 1:length(subject_nums)
    subject_num = subject_nums{i}
    roi_dir = ([getenv('SUBJECTS_DIR') '/vaegan-sub-0' num2str(subject_num) '-all/roi/']);
    correlations_dir = [getenv('FUNCTIONALS_DIR') '/vaegan-consolidated/unpackdata/vaegan-sub-0' num2str(subject_num) '-all/bold/correlations/'];
    
    % Get the left and right betas so that we can use the metadata and
    % overwrite the vol with the correlations
    left_hemi_beta_dir = [getenv('FUNCTIONALS_DIR') '/vaegan-consolidated/unpackdata/vaegan-sub-0' num2str(subject_num) '-all/bold/' getenv('MODEL_NAME') '.lh'];
    right_hemi_beta_dir = [getenv('FUNCTIONALS_DIR') '/vaegan-consolidated/unpackdata/vaegan-sub-0' num2str(subject_num) '-all/bold/' getenv('MODEL_NAME') '.rh'];
    left_hemi_betas = MRIread([left_hemi_beta_dir '/beta.nii.gz']);
    right_hemi_betas = MRIread([right_hemi_beta_dir '/beta.nii.gz']);
    
    for j = 1:length(rois)
        roi = rois{j};
        correlation = load([correlations_dir getenv('MODEL_NAME') '.' roi '.correlations.mat']).data;
        left_roi = load([roi_dir 'l' roi '.surf.thresholded.mat']).threshold_roi;
        right_roi = load([roi_dir 'r' roi '.surf.thresholded.mat']).threshold_roi;
        
        left_correlations = correlation(1:sum(left_roi));
        right_correlations = correlation(sum(left_roi)+1:length(correlation));
        
        expanded_left_correlations = left_roi;
        expanded_left_correlations(find(left_roi)) = left_correlations;
        expanded_right_correlations = right_roi;
        expanded_right_correlations(find(right_roi)) = right_correlations;
        
        
        left_hemi_betas.vol = expanded_left_correlations;
        left_hemi_betas.fspec = [correlations_dir getenv('MODEL_NAME') '.l' roi '.correlations.nii.gz'];
        MRIwrite(left_hemi_betas, left_hemi_betas.fspec);
        
        right_hemi_betas.vol = expanded_right_correlations;
        right_hemi_betas.fspec = [correlations_dir getenv('MODEL_NAME') '.r' roi '.correlations.nii.gz'];
        MRIwrite(right_hemi_betas, right_hemi_betas.fspec);
    end
    
    
    
    left_hemi_beta_dir = [getenv('FUNCTIONALS_DIR') '/vaegan-consolidated/unpackdata/vaegan-sub-0' num2str(subject_num) '-all/bold/' getenv('MODEL_NAME') '.lh'];
    right_hemi_beta_dir = [getenv('FUNCTIONALS_DIR') '/vaegan-consolidated/unpackdata/vaegan-sub-0' num2str(subject_num) '-all/bold/' getenv('MODEL_NAME') '.rh'];
    
    left_hemi_betas = MRIread([left_hemi_beta_dir '/beta.nii.gz']);
    right_hemi_betas = MRIread([right_hemi_beta_dir '/beta.nii.gz']);
    
    left_hemi_labels = repmat({'lh'}, 1, left_hemi_betas.nvoxels);
    right_hemi_labels = repmat({'rh'}, 1, right_hemi_betas.nvoxels);
    hemi_labels = cat(2, left_hemi_labels, right_hemi_labels);
    
    % The two represents the face bias betas and the oneback betas which
    % come between the latent dimensions and the test images
    left_hemi_betas = squeeze(left_hemi_betas.vol(1,:,1,1:str2num(getenv('LATENT_DIMENSIONS'))+str2num(getenv('NUM_TEST_IMAGES'))+2));
    right_hemi_betas = squeeze(right_hemi_betas.vol(1,:,1,1:str2num(getenv('LATENT_DIMENSIONS'))+str2num(getenv('NUM_TEST_IMAGES'))+2));
    
    betas = [left_hemi_betas; right_hemi_betas];
    save([getenv('FUNCTIONALS_DIR') '/vaegan-consolidated/unpackdata/vaegan-sub-0' num2str(subject_num) '-all/bold/' getenv('MODEL_NAME') '.betas.mat'], 'betas', 'hemi_labels')
end