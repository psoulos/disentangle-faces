assert(~(getenv('FIELDTRIP_DIR') == ""), 'You must first set the environment variable FIELDTRIP_DIR')
assert(~(getenv('SUBJECTS_DIR') == ""), 'You must first set the environment variable SUBJECTS_DIR')
assert(~(getenv('FUNCTIONALS_DIR') == ""), 'You must first set the environment variable FUNCTIONALS_DIR')

addpath([getenv('FIELDTRIP_DIR/') '/external/freesurfer'])
subject_nums = {1 2 3 4};
hemis = {'l' 'r'};
rois = {'FFA' 'OFA' 'STS'};
for i = 1:length(subject_nums)
    subject_num = subject_nums{1}
    roi_dir = ([getenv('SUBJECTS_DIR') '/vaegan-sub-0' num2str(subject_num) '-all/roi'])
    for j = 1:length(rois)
        roi = rois{j}
        for k = 1:length(hemis)
           hemi = hemis{k}
           roi_surface = MRIread([roi_dir '/' hemi roi '.surf.nii.gz']);
           significance = MRIread([getenv('SUBJECTS_DIR') '/vaegan-consolidated/unpackdata/vaegan-sub-0' num2str(subject_num) '-all/bold/vaegan-localizer-sm5-' hemi 'h/faces-constrast-objects/sig.nii.gz']);
           roi_significance = significance.vol;
           roi_significance(roi_surface.vol == 0) = 0;
           num_voxels = sum(roi_surface.vol)
           num_top10 = floor(num_voxels / 10)
           [sorted_roi_significance, indices] = sort(roi_significance, 'descend');
           threshold_roi = roi_surface.vol;
           threshold_roi(indices(num_top10+1:end)) = 0;
           
           roi_surface.vol = threshold_roi;
           roi_surface.fspec = [roi_dir '/' hemi roi '.surf.thresholded.nii.gz'];
           
           MRIwrite(roi_surface, roi_surface.fspec)
        end
    end
end