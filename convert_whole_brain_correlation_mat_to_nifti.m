assert(~(getenv('FIELDTRIP_DIR') == ""), 'You must first set the environment variable FIELDTRIP_DIR')
assert(~(getenv('FUNCTIONALS_DIR') == ""), 'You must first set the environment variable FUNCTIONALS_DIR')
assert(~(getenv('SUBJECTS_DIR') == ""), 'You must first set the environment variable SUBJECTS_DIR')
assert(~(getenv('MODEL_NAME') == ""), 'You must first set the environment variable MODEL_NAME')
assert(~(getenv('SCORE_THRESHOLD') == ""), 'You must first set the environment variable SCORE_THRESHOLD')


addpath([getenv('FIELDTRIP_DIR') '/external/freesurfer'])
subject_nums = {1 2 3 4};
hemis = {'l' 'r'};
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

    correlation = load([correlations_dir getenv('MODEL_NAME') '.score.whole.correlations.mat']);
    correlation = correlation.data;

    left_localizer = load([roi_dir 'whole_brain_score_' getenv('SCORE_THRESHOLD') '.lh.surf.thresholded.mat']);
    left_localizer = left_localizer.left_score > 0;
    right_localizer = load([roi_dir 'whole_brain_score_' getenv('SCORE_THRESHOLD') '.rh.surf.thresholded.mat']);
    right_localizer = right_localizer.right_score > 0;

    left_correlations = correlation(1:sum(left_localizer));
    right_correlations = correlation(sum(left_localizer)+1:length(correlation));

    % We need to convert the logical array into a numerical array with 'double'
    expanded_left_correlations = double(left_localizer);
    expanded_left_correlations(find(left_localizer)) = left_correlations;
    expanded_right_correlations = double(right_localizer);
    expanded_right_correlations(find(right_localizer)) = right_correlations;

    left_hemi_betas.vol = expanded_left_correlations;
    left_hemi_betas.fspec = [correlations_dir getenv('MODEL_NAME') '.score.' getenv('SCORE_THRESHOLD') '.lh.correlations.nii.gz'];
    MRIwrite(left_hemi_betas, left_hemi_betas.fspec);

    right_hemi_betas.vol = expanded_right_correlations;
    right_hemi_betas.fspec = [correlations_dir getenv('MODEL_NAME') '.score.' getenv('SCORE_THRESHOLD') '.rh.correlations.nii.gz'];
    MRIwrite(right_hemi_betas, right_hemi_betas.fspec);
end