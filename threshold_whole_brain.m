assert(~(getenv('FIELDTRIP_DIR') == ""), 'You must first set the environment variable FIELDTRIP_DIR')
assert(~(getenv('SUBJECTS_DIR') == ""), 'You must first set the environment variable SUBJECTS_DIR')
assert(~(getenv('FUNCTIONALS_DIR') == ""), 'You must first set the environment variable FUNCTIONALS_DIR')
assert(~(getenv('THRESHOLD_PARAMETER') == ""), 'You must first set the environment variable THRESHOLD_PARAMETER')

threshold_parameter = str2num(getenv('THRESHOLD_PARAMETER'));
addpath([getenv('FIELDTRIP_DIR') '/external/freesurfer'])
subject_nums = {1 2 3 4};
for i = 1:length(subject_nums)
    subject_num = subject_nums{i}
    roi_dir = ([getenv('SUBJECTS_DIR') '/vaegan-sub-0' num2str(subject_num) '-all/roi'])

    % blend of reliability and localizer
    % Z score first
    % Combine them
    % take the top 10 percent or some other threshold (2xstd above mean)
    
    left_localizer = MRIread([getenv('FUNCTIONALS_DIR') '/vaegan-consolidated/unpackdata/vaegan-sub-0' num2str(subject_num) '-all/bold/vaegan-localizer-sm5-lh/faces-constrast-objects/sig.nii.gz']);
    right_localizer = MRIread([getenv('FUNCTIONALS_DIR') '/vaegan-consolidated/unpackdata/vaegan-sub-0' num2str(subject_num) '-all/bold/vaegan-localizer-sm5-rh/faces-constrast-objects/sig.nii.gz']);
    whole_brain_localizer = cat(2,left_localizer.vol, right_localizer.vol);
    whole_brain_localizer = normalize(whole_brain_localizer);
    
    left_reliability = MRIread([getenv('FUNCTIONALS_DIR') '/vaegan-consolidated/unpackdata/vaegan-sub-0' num2str(subject_num) '-all/bold/correlations/vgg.fc7.24.split_test.lwhole_brain.correlations.nii.gz']);
    right_reliability = MRIread([getenv('FUNCTIONALS_DIR') '/vaegan-consolidated/unpackdata/vaegan-sub-0' num2str(subject_num) '-all/bold/correlations/vgg.fc7.24.split_test.rwhole_brain.correlations.nii.gz']);
    whole_brain_reliability = cat(2,left_reliability.vol, right_reliability.vol);
    whole_brain_reliability = normalize(whole_brain_reliability);

    whole_brain_score = whole_brain_localizer + whole_brain_reliability;
    
    %num_voxels = size(whole_brain_score, 2)
    %num_top10 = floor(num_voxels / 10)
    %[sorted_whole_brain_score, indices] = sort(whole_brain_score, 'descend');
    %whole_brain_score(indices(num_top10+1:end)) = 0;
    
    threshold = threshold_parameter * std(whole_brain_score);
    whole_brain_score(whole_brain_score < threshold) = 0;
    
    left_score = whole_brain_score(1:size(left_localizer.vol, 2));
    right_score = whole_brain_score(size(left_localizer.vol, 2)+1:end);

    left_reliability.vol = left_score;
    left_reliability.fspec = [roi_dir '/whole_brain_score_' num2str(threshold_parameter) '.lh.surf.thresholded.nii.gz'];
    MRIwrite(left_reliability, left_reliability.fspec);
    save([roi_dir '/whole_brain_score_' num2str(threshold_parameter) '.lh.surf.thresholded.mat'], 'left_score');
    
    right_reliability.vol = right_score;
    right_reliability.fspec = [roi_dir '/whole_brain_score_' num2str(threshold_parameter) '.rh.surf.thresholded.nii.gz'];
    MRIwrite(right_reliability, right_reliability.fspec);
    save([roi_dir '/whole_brain_score_' num2str(threshold_parameter) '.rh.surf.thresholded.mat'], 'right_score');
end