addpath('corr_col.m')
assert(~(getenv('FUNCTIONALS_DIR') == ""), 'You must first set the environment variable FUNCTIONALS_DIR')
assert(~(getenv('SUBJECTS_DIR') == ""), 'You must first set the environment variable SUBJECTS_DIR')
assert(~(getenv('FIELDTRIP_DIR') == ""), 'You must first set the environment variable FIELDTRIP_DIR')

addpath([getenv('FIELDTRIP_DIR') '/external/freesurfer'])

%% Models
models = {'factor_vae.latent_24.hyper_10.random_50751608.train_bias', ...
    'vae.latent_24.hyper_1.random_50639474.train_bias', ...
    'vgg.fc7.24.train_bias'};

encoding_corr = cell(4,1);
encoding_corr_feat = cell(4,1);
mkdir('encoding_corr_feat')

for model_i = 1:length(models)
    model = models{model_i}
    for s = 1:4
        fprintf('Subject %i\n', s)
        %% ROIs
        roi_dir = ([getenv('SUBJECTS_DIR') '/vaegan-sub-0' num2str(s) '-all/roi/']);
        bold_dir = [getenv('FUNCTIONALS_DIR') '/vaegan-consolidated/unpackdata/vaegan-sub-0' num2str(s) '-all/bold/'];

        % Read the beta nifti so we can use that data structure to save our
        % preference map
        right_hemi_betas = MRIread([bold_dir model '.rh/beta.nii.gz']);
        right_hemi_all_rois = zeros(1, right_hemi_betas.nvoxels);
        
        left_hemi_betas = MRIread([bold_dir model '.lh/beta.nii.gz']);
        left_hemi_all_rois = zeros(1, left_hemi_betas.nvoxels);

        %% Load betas
        beta_file = [bold_dir '/' model '.betas.mat'];
        load(beta_file)

        rh_localizer = [roi_dir 'whole_brain_score_1.5.rh.surf.thresholded.mat'];
        lh_localizer = [roi_dir 'whole_brain_score_1.5.lh.surf.thresholded.mat'];

        left_localizer = load(lh_localizer);
        right_localizer = load(rh_localizer);

        localizer = [left_localizer.left_score > 0 right_localizer.right_score > 0];
 
        W = betas(localizer>0,1:24);
        b = betas(localizer>0,25);
        y = betas(localizer>0,27:end);

        %% Extract test image latent values
        img_latent_file = [bold_dir '/correlations/' model ...
         '.output.mat'];
        load(img_latent_file);
        test_img_list = {{'M2553', 'F1631', 'M2424', 'F1235', 'F1148', 'M2156', 'F2376', ...
            'M1584', 'M2466', 'F2068', 'F1586', 'F1232', 'M2203', 'M1365',...
            'M2248', 'F2467', 'M2336', 'F1145', 'F2377', 'M2246'}, ...

            {'M7260', 'M8712', 'M7704', 'F7216', 'F8408', 'F6117', 'F7792',...
            'F8669', 'M4446', 'M4535', 'M6338', 'M7041', 'F5414', 'M6776', ...
            'M4621', 'F4622', 'F6118', 'M6953', 'F5724', 'F6116'}, ...

            {'M10035', 'F09021', 'M12366', 'M10124', 'M11003', 'F09903',...
            'F12323', 'F10912', 'M10165', 'F09152', 'F11266', 'M10783', ...
            'F09109', 'M12233', 'F11440', 'M13160', 'F11400', 'F08933',...
             'M08800', 'M11927'},...

            {'F14697', 'F14081', 'M14039', 'F13996', 'M17160', 'F15049', ...
            'F15137', 'M13644', 'M15665', 'M13289', 'M15488', 'M17204',...
            'F15976', 'M16368', 'F13774', 'F15404', 'M17336', 'F13640',...
            'F14520', 'M14256'}};

        X = zeros(24,20);
        gender = zeros(1,20);
        for i = 1:length(test_img_list{s})
            X(:,i) = eval(test_img_list{s}{i})';
            gender(i) = num2str(test_img_list{s}{i}(1));
            % 70 is F
            % 77 is M
        end

        y_hat = W*X+b;

        %% encoding
        encoding_corr{s,1}(:) = corr_col(y',y_hat');

        %% feature mapping
        for f = 1:size(W,2)
            W_f = zeros(size(W));
            W_f(:,f) = W(:,f);
            y_hat_f = W_f*X+b;
            encoding_corr_feat{s,1}(:,f) = corr_col(y',y_hat_f');
        end

        [max_corr, index] = max(encoding_corr_feat{s,1}, [], 2);
        
        num_left_voxels = sum(left_localizer.left_score > 0);
        left_hemi_all_rois(find(left_localizer.left_score)) = index(1:num_left_voxels);
        right_hemi_all_rois(find(right_localizer.right_score)) = index(num_left_voxels+1:end);

        % mean(encoding_corr_feat{1,1}) to get the average prediction
        % accuracy for each latent dimension to an ROI

        writematrix(encoding_corr_feat{s,1}, ['encoding_corr_feat/subject_' num2str(s) '_roi_score_model_' model '.csv'])

        mkdir([bold_dir 'preference_maps/'])
        right_hemi_betas.vol = right_hemi_all_rois;
        right_hemi_betas.fspec = [bold_dir 'preference_maps/' model '.score.right.preference.nii.gz'];
        MRIwrite(right_hemi_betas, right_hemi_betas.fspec);
        
        left_hemi_betas.vol = left_hemi_all_rois;
        left_hemi_betas.fspec = [bold_dir 'preference_maps/' model '.score.left.preference.nii.gz'];
        MRIwrite(left_hemi_betas, left_hemi_betas.fspec);
    end
end
