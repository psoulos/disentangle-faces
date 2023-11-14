addpath('corr_col.m')
assert(~(getenv('FUNCTIONALS_DIR') == ""), 'You must first set the environment variable FUNCTIONALS_DIR')
assert(~(getenv('SUBJECTS_DIR') == ""), 'You must first set the environment variable SUBJECTS_DIR')
assert(~(getenv('FIELDTRIP_DIR') == ""), 'You must first set the environment variable FIELDTRIP_DIR')

addpath([getenv('FIELDTRIP_DIR') '/external/freesurfer'])

%% Models
models = {'factor_vae.latent_24.hyper_10.random_50751608.train_bias'}
%models = {'factor_vae.latent_24.hyper_10.random_50751608.train_bias', ...
%    'vae.latent_24.hyper_1.random_50639474.train_bias', ...
%    'vgg.fc7.24.train_bias'};
%% Localizer
ROIs = {'FFA', 'OFA', 'STS'};
n_rois = 3;
n_subjects = 4;
n_dimensions = 24;
mean_pairwise = zeros(n_subjects, length(models));
mean_pairwise_VR = mean_pairwise;
n_way_acc = mean_pairwise;
encoding_corr = cell(4,3);
encoding_corr_feat = cell(4,3);
mkdir('encoding_corr_feat')

permutation_test = true
if permutation_test
    n_permutations = 5000;
    n_test_images = 20;
    permutation_matrix = zeros(n_permutations, n_test_images);
    permutation_correlations = zeros(n_subjects, n_dimensions, n_rois, n_permutations);

    % Pre-calculate the permutation matrix so that I can share it between the
    % ROIs
    for i = 1:n_permutations
        permutation_matrix(i,:) = randperm(20);
    end
end

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

        %% Load betas
        beta_file = [bold_dir '/' model '.betas.mat'];
        load(beta_file)

        for r = 1:length(ROIs)
            roi = ROIs{r};
            fprintf('ROI %s\n', roi)
            rroi = ['r' ROIs{r}];%'whole_brain_score_1.5.rh';
            %lroi = ['l' ROIs{r}];%'whole_brain_score_1.5.lh';
            %lh_localizer = [roi_dir lroi '.surf.thresholded.both.mat'];
            rh_localizer = [roi_dir rroi '.surf.thresholded.both.mat'];

            %load(lh_localizer)
            load(rh_localizer)

            if roi 
                localizer = zeros(1,size(betas,1));
                localizer(end-length(threshold_roi)+1:end) = threshold_roi;
            else
                % TODO
                localizer = [left_score right_score];
            end

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
            encoding_corr{s,r}(:) = corr_col(y',y_hat');

            %% feature mapping
            for f = 1:size(W,2)
                W_f = zeros(size(W));
                W_f(:,f) = W(:,f);
                y_hat_f = W_f*X+b;
                encoding_corr_feat{s,r}(:,f) = corr_col(y',y_hat_f');
                if permutation_test
                    for i_permutation = 1:n_permutations
                        shuffled_y = zeros(size(y));
                        for i_voxel = 1:size(y, 1)
                            shuffled_y(i_voxel, :) = y(i_voxel, permutation_matrix(i_permutation, :));
                        end
                        % TODO: permutation_correlations needs an ROI
                        % dimension
                        permutation_correlations(s, f, r, i_permutation) = mean(corr_col(shuffled_y', y_hat_f'));
                    end
                end
            end

            [max_corr, index] = max(encoding_corr_feat{s,r}, [], 2);
            right_hemi_all_rois(find(threshold_roi)) = index;

            % mean(encoding_corr_feat{1,1}) to get the average prediction
            % accuracy for each latent dimension to an ROI

            writematrix(encoding_corr_feat{s,r}, ['encoding_corr_feat/subject_' num2str(s) '_roi_' roi '_model_' model '.csv'])
        end

        mkdir([bold_dir 'preference_maps/'])
        right_hemi_betas.vol = right_hemi_all_rois;
        right_hemi_betas.fspec = [bold_dir 'preference_maps/' model '.right.preference.nii.gz'];
        MRIwrite(right_hemi_betas, right_hemi_betas.fspec);
    end
    
    % I put the correlations in a cell instead of a tensor before, so put
    % the data in a tensor for easier processing
    subj_dim_roi_correlations = zeros(n_subjects, n_dimensions, n_rois);
    for s = 1:4
        for r = 1:length(ROIs)
            for d = 1:n_dimensions
                subj_dim_roi_correlations(s, d, r) = mean(encoding_corr_feat{s,r}(:, d));
            end
        end
    end

    
    % Mean the results across subjects, iterate over the ROIs, latent
    % dimensions and count the times the real correlation is greater than
    % the permutations
    % By ROI by dimension versus chance
    mean_permutation_correlations = mean(permutation_correlations, 1);
    mean_dim_roi_correlations = mean(subj_dim_roi_correlations, 1);
    for r = 1:length(ROIs)
        for d = 1:n_dimensions
            actual_less_than_permutation = sum(mean_dim_roi_correlations(1, d, r) < mean_permutation_correlations(1,d,r,:));
            p = (actual_less_than_permutation+1)/n_permutations;
            fprintf('ROI %s, Dim %d, p=%f, sig=%d\n', ROIs{r}, d, p, p<.05)
        end
    end
    
    % ROI vs ROI
    for d = 1:n_dimensions
        actual_ffa_minus_ofa = mean_dim_roi_correlations(1, d, 1) - mean_dim_roi_correlations(1, d, 2);
        permutation_ffa_minus_ofa = mean_permutation_correlations(1, d, 1, :) - mean_permutation_correlations(1, d, 2, :);
        actual_less_than_permutation = sum(actual_ffa_minus_ofa < permutation_ffa_minus_ofa);
        p = (actual_less_than_permutation+1)/n_permutations;
        fprintf('FFA-OFA, Dim %d, p=%f, p<.05=%d, p>.95=%d\n', d, p, p<.05, p>.95)
    end
end
