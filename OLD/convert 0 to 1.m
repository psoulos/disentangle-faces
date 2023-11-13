assert(~(getenv('FUNCTIONALS_DIR') == ""), 'You must first set the environment variable FUNCTIONALS_DIR')
assert(~(getenv('SUBJECTS_DIR') == ""), 'You must first set the environment variable SUBJECTS_DIR')
assert(~(getenv('FIELDTRIP_DIR') == ""), 'You must first set the environment variable FIELDTRIP_DIR')

addpath([getenv('FIELDTRIP_DIR') '/external/freesurfer'])

models = {'factor_vae.latent_24.hyper_10.random_50751608.train_bias'};

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
            rroi = ['r' ROIs{r}];
            
            rh_localizer = [roi_dir rroi '.surf.thresholded.both.mat'];
            load(rh_localizer);
            
            localizer = zeros(1,size(betas,1));
            localizer(end-length(threshold_roi)+1:end) = threshold_roi;
            
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
            for i = 1:length(test_img_list{s})
                X(:,i) = eval(test_img_list{s}{i})';
            end
            
            w_inv = pinv(W);
            x_hat = w_inv*(y-b);
            
            all_images_correct = 0;
            number_of_pairwise_comparison = 0;
            for i = 1:size(x_hat,2)
                reconstruction = x_hat(:,i);
                ground_truth = X(:,i);
                recon_ground_truth_corr = corr(reconstruction, ground_truth, 'Type','Spearman');%corr_col(reconstruction, ground_truth);
                corrs = zeros(size(x_hat,2),1);
                for j = 1:size(x_hat,2)
                    distractor = X(:,j);
                    distractor_corr = corr(reconstruction, distractor, 'Type','Spearman');
                    corrs(j) = distractor_corr;
                end
                total_correct = sum(recon_ground_truth_corr > corrs);
                total = length(corrs)-1;
                all_images_correct = all_images_correct + total_correct;
                number_of_pairwise_comparison = number_of_pairwise_comparison + total;
                %fprintf('Subject %i Test Image %i %i/%i\n', s, i, total_correct, total)
            end
            fprintf('Subject %i %i/%i\n', s, all_images_correct, number_of_pairwise_comparison)
        end
    end
end
