addpath('~/repos/fieldtrip/external/freesurfer/')
data_type = 'sig';
data_folder = ['~/Downloads/' data_type '/'];

subj = {'sub-01', 'sub-02', 'sub-03', 'sub-04'};

for s = 1:length(subj)
    
    RH = MRIread([data_folder subj{s} '/rh.sig.nii.gz']);
    LH = MRIread([data_folder subj{s} '/lh.sig.nii.gz']);
    
   	hemi = [repmat({'rh'}, size(RH.vol)) repmat({'lh'}, size(LH.vol))];
    data = [RH.vol, LH.vol];
    
    save([subj{s} '_' data_type '.mat'], 'hemi', 'data')
end