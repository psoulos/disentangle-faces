assert(~(getenv('FIELDTRIP_DIR') == ""), 'You must first set the environment variable FIELDTRIP_DIR')
assert(~(getenv('ALIGNMENT_DIR') == ""), 'You must first set the environment variable ALIGNMENT_DIR')
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
    alignment_dir = getenv('ALIGNMENT_DIR');
    
    % Get the left and right betas so that we can use the metadata and
    % overwrite the vol with the alignments
    left_hemi_beta_dir = [getenv('FUNCTIONALS_DIR') '/vaegan-consolidated/unpackdata/vaegan-sub-0' num2str(subject_num) '-all/bold/' getenv('MODEL_NAME') '.lh'];
    right_hemi_beta_dir = [getenv('FUNCTIONALS_DIR') '/vaegan-consolidated/unpackdata/vaegan-sub-0' num2str(subject_num) '-all/bold/' getenv('MODEL_NAME') '.rh'];
    left_hemi_betas = MRIread([left_hemi_beta_dir '/beta.nii.gz']);
    right_hemi_betas = MRIread([right_hemi_beta_dir '/beta.nii.gz']);
    
    left_hemi_all_rois = zeros(1, left_hemi_betas.nvoxels);
    right_hemi_all_rois = zeros(1, right_hemi_betas.nvoxels);
    
    for j = 1:length(rois)
        roi = rois{j}
        alignment = load([alignment_dir '/subject' num2str(subject_num) '.' getenv('MODEL_NAME') '.' roi '.alignments.mat']);
        alignment = alignment.data;
        left_roi = load([roi_dir 'l' roi '.surf.thresholded.both.mat']);
        left_roi = left_roi.threshold_roi;
        right_roi = load([roi_dir 'r' roi '.surf.thresholded.both.mat']);
        right_roi = right_roi.threshold_roi;
        
        left_alignments = alignment(1:sum(left_roi));
        right_alignments = alignment(sum(left_roi)+1:length(alignment));
        
        expanded_left_alignments = left_roi;
        expanded_left_alignments(find(left_roi)) = left_alignments;
        expanded_right_alignments = right_roi;
        expanded_right_alignments(find(right_roi)) = right_alignments;
        
        
        left_hemi_betas.vol = expanded_left_alignments;
        left_hemi_betas.fspec = [alignment_dir '/subject' num2str(subject_num) '.' getenv('MODEL_NAME') '.l' roi '.alignments.nii.gz'];
        MRIwrite(left_hemi_betas, left_hemi_betas.fspec);
        
        right_hemi_betas.vol = expanded_right_alignments;
        right_hemi_betas.fspec = [alignment_dir '/subject' num2str(subject_num) '.' getenv('MODEL_NAME') '.r' roi '.alignments.nii.gz'];
        MRIwrite(right_hemi_betas, right_hemi_betas.fspec);
        
        left_hemi_all_rois = left_hemi_all_rois + expanded_left_alignments;
        right_hemi_all_rois = right_hemi_all_rois + expanded_right_alignments;
    end
    
    left_hemi_betas.vol = left_hemi_all_rois;
    left_hemi_betas.fspec = [alignment_dir '/subject' num2str(subject_num) '.' getenv('MODEL_NAME') '.left.alignments.nii.gz'];
    MRIwrite(left_hemi_betas, left_hemi_betas.fspec);
    
    right_hemi_betas.vol = right_hemi_all_rois;
    right_hemi_betas.fspec = [alignment_dir '/subject' num2str(subject_num) '.' getenv('MODEL_NAME') '.right.alignments.nii.gz'];
    MRIwrite(right_hemi_betas, right_hemi_betas.fspec);
end