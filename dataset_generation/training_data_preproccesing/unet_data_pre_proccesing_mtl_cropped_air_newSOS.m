clear
clc

LoadData = sprintf('/kwave_results/wheat/raw/wroot1_example1.mat');
load(LoadData);

LoadData = sprintf('/wheat_processed/preprocessing_order.mat');
load(LoadData);
dt = 1/Fs;
t_end = size(data,2)*dt;
t = 0:dt:t_end-dt;
timeGain = t./t_end; % TGC


%random_example_vector = randperm(10000);
%example_vector_saved = random_example_vector;

 
%% Training Data

for train = 1:9000
        
         % Loading saved k-wave data
         LoadData = sprintf('/kwave_results/wheat/raw/wroot*_example%d.mat',example_vector_saved(train));
         matfile = dir(LoadData);
         load(fullfile(matfile.folder,matfile.name));
         saveImage = sprintf('%d.png',train);
                 
         % Extracting and Plotting  SoS Map 
         [a, b] = size(SOSmap); 
         crop_axial = round(a/3); %data to be removed from the bottom
         crop_lateral = round(b/6); %1/6th on each side, so a third in total - 45cm down to 30cm
         SOSmap_crop = SOSmap(1:(a - crop_axial), crop_lateral: (b - crop_lateral));
         SOSmap_crop8 = (SOSmap_crop-340)./460;
         SOSmap_256 = imresize(SOSmap_crop8, [256 256]);  % resizing to 256x256
         imwrite(SOSmap_256, strcat('/wheat_processed/train/sos/', saveImage));
        
         % Extracting and Plotting Reconstructed Image
         [a, b] = size(roots2); 
         crop_axial = round(a/3); %data to be removed from the bottom
         crop_lateral = round(b/6); %1/6th on each side, so a third in total - 45cm down to 30cm
         ReconImage_crop = roots2(1:(a - crop_axial), crop_lateral: (b - crop_lateral));
         Recon_abs = abs(ReconImage_crop);
         Recon_gray = mat2gray(Recon_abs,[min(Recon_abs(:)) max(Recon_abs(:))]);
         Recon_256 = round(imresize(Recon_gray, [256 256]));
         imwrite(Recon_256, strcat('/wheat_processed/train/img_bin/', saveImage));
         
         
         % Extracting and Plotting Reconstructed Image
         [a, b] = size(ReconImage); 
         crop_axial = round(a/3); %data to be removed from the bottom
         crop_lateral = round(b/6); %1/6th on each side, so a third in total - 45cm down to 30cm
         ReconImage_crop = ReconImage(1:(a - crop_axial), crop_lateral: (b - crop_lateral));
         Recon_abs = abs(ReconImage_crop);
         Recon_gray = mat2gray(Recon_abs,[min(Recon_abs(:)) max(Recon_abs(:))]);
         Recon_clean = zeros(size(Recon_abs));
         Recon_clean(28:end, :) = Recon_abs(28:end,:);
         Recon_clean_gray = mat2gray(Recon_clean,[min(Recon_clean(:)) max(Recon_clean(:))]);
         Recon_256 = imresize(Recon_gray, [256 256]);
         Recon_256_clean = imresize(Recon_clean_gray, [256 256]);
         imwrite(Recon_256, strcat('/wheat_processed/train/img/', saveImage));
         imwrite(Recon_256_clean, strcat('/wheat_processed/train/clean/img/', saveImage));
         
         % Extracting and Plotting Reconstructed Image - 500 m/s
         ReconImage_crop_500 = ReconImage_500(1:(a - crop_axial), crop_lateral: (b - crop_lateral));
         Recon_abs_500 = abs(ReconImage_crop_500);
         Recon_gray_500 = mat2gray(Recon_abs_500,[min(Recon_abs_500(:)) max(Recon_abs_500(:))]);
         Recon_clean_500 = zeros(size(Recon_abs_500));
         Recon_clean_500(28:end, :) = Recon_abs_500(28:end,:);
         Recon_clean_gray_500 = mat2gray(Recon_clean_500,[min(Recon_clean_500(:)) max(Recon_clean_500(:))]);
         Recon_256_500 = imresize(Recon_gray_500, [256 256]);
         Recon_256_clean_500 = imresize(Recon_clean_gray_500, [256 256]);
         imwrite(Recon_256_500, strcat('/wheat_processed/train/img_500/', saveImage));
         imwrite(Recon_256_clean_500, strcat('/wheat_processed/train/clean/img_500/', saveImage));
         
         % Extracting and Plotting Reconstructed Image - 650 m/s
         ReconImage_crop_650 = ReconImage_650(1:(a - crop_axial), crop_lateral: (b - crop_lateral));
         Recon_abs_650 = abs(ReconImage_crop_650);
         Recon_gray_650 = mat2gray(Recon_abs_650,[min(Recon_abs_650(:)) max(Recon_abs_650(:))]);
         Recon_clean_650 = zeros(size(Recon_abs_650));
         Recon_clean_650(28:end, :) = Recon_abs_650(28:end,:);
         Recon_clean_gray_650 = mat2gray(Recon_clean_650,[min(Recon_clean_650(:)) max(Recon_clean_650(:))]);
         Recon_256_650 = imresize(Recon_gray_650, [256 256]);
         Recon_256_clean_650 = imresize(Recon_clean_gray_650, [256 256]);
         imwrite(Recon_256_650, strcat('/wheat_processed/train/img_650/', saveImage));
         imwrite(Recon_256_clean_650, strcat('/wheat_processed/train/clean/img_650/', saveImage));
     
         % Extracting and Plotting Reconstructed Image - 800 m/s
         ReconImage_crop_800 = ReconImage_800(1:(a - crop_axial), crop_lateral: (b - crop_lateral));
         Recon_abs_800 = abs(ReconImage_crop_800);
         Recon_gray_800 = mat2gray(Recon_abs_800,[min(Recon_abs_800(:)) max(Recon_abs_800(:))]);
         Recon_clean_800 = zeros(size(Recon_abs_800));
         Recon_clean_800(28:end, :) = Recon_abs_800(28:end,:);
         Recon_clean_gray_800 = mat2gray(Recon_clean_800,[min(Recon_clean_800(:)) max(Recon_clean_800(:))]);
         Recon_256_800 = imresize(Recon_gray_800, [256 256]);
         Recon_256_clean_800 = imresize(Recon_clean_gray_800, [256 256]);
         imwrite(Recon_256_800, strcat('/wheat_processed/train/img_800/', saveImage));
         imwrite(Recon_256_clean_800, strcat('/wheat_processed/train/clean/img_800/', saveImage));
         
         % Combining 500, 650 and 800 to one unfocused image
         ReconImage_abs = cat(3, Recon_abs_500, Recon_abs_650, Recon_abs_800);
         Recon_gray = mat2gray(ReconImage_abs,[min(ReconImage_abs(:)) max(ReconImage_abs(:))]);
         Recon_256_uimg = imresize(Recon_gray, [256 256]);
         
         ReconImage_abs_clean = cat(3, Recon_clean_500, Recon_clean_650, Recon_clean_800);
         Recon_gray_clean = mat2gray(ReconImage_abs_clean,[min(ReconImage_abs_clean(:)) max(ReconImage_abs_clean(:))]);
         Recon_256_uimg_clean = imresize(Recon_gray_clean, [256 256]);
         
         imwrite(Recon_256_uimg, strcat('/wheat_processed/train/uimg/', saveImage));
         imwrite(Recon_256_uimg_clean, strcat('/wheat_processed/train/clean/uimg/', saveImage));
         
         % Extracting and Plotting A-Scan Data
         t_end_new = t_end*sqrt(((a - crop_axial)^2 + (b-crop_lateral)^2)/(a^2 + b^2));
         data_crop = data_down(:,1:round(t_end_new*Fs));
         data_TGC = data_crop(1:end,:).*sqrt(timeGain(1:size(data_crop,1)))';
         Ascan_gray = mat2gray(data_TGC, [-1*max(abs(data_TGC(:))) max(abs(data_TGC(:)))]); % converting to [0,1]
         Ascan_256 = imresize(Ascan_gray, [256 256]); %resampling and resizing to 256x256
         imwrite(Ascan_256', strcat('/wheat_processed/train/ascan/', saveImage));
        
         
end

%% Validation Data

ctr = 1;
for val = 9001:9500
        
         % Loading saved k-wave data
         LoadData = sprintf('/kwave_results/wheat/raw/wroot*_example%d.mat',example_vector_saved(val));
         matfile = dir(LoadData);
         load(fullfile(matfile.folder,matfile.name));
         saveImage = sprintf('%d.png',ctr);
         

         % Extracting and Plotting Reconstructed Image
         [a, b] = size(roots2); 
         crop_axial = round(a/3); %data to be removed from the bottom
         crop_lateral = round(b/6); %1/6th on each side, so a third in total - 45cm down to 30cm
         ReconImage_crop = roots2(1:(a - crop_axial), crop_lateral: (b - crop_lateral));
         Recon_abs = abs(ReconImage_crop);
         Recon_gray = mat2gray(Recon_abs,[min(Recon_abs(:)) max(Recon_abs(:))]);
         Recon_256 = round(imresize(Recon_gray, [256 256]));
         imwrite(Recon_256, strcat('/wheat_processed/val/img_bin/', saveImage));
         
         %  Extracting and Plotting corrected 8 bit SoS Map 
         [a, b] = size(SOSmap); 
         crop_axial = round(a/3); %data to be removed from the bottom
         crop_lateral = round(b/6); %1/6th on each side, so a third in total - 45cm down to 30cm
         SOSmap_crop = SOSmap(1:(a - crop_axial), crop_lateral: (b - crop_lateral));
         SOSmap_crop8 = (SOSmap_crop-340)./460;
         SOSmap_256 = imresize(SOSmap_crop8, [256 256]);  % resizing to 256x256
         imwrite(SOSmap_256, strcat('/wheat_processed/val/sos/', saveImage));
        
         % Extracting and Plotting Reconstructed Image
         [a, b] = size(ReconImage); 
         crop_axial = round(a/3); %data to be removed from the bottom
         crop_lateral = round(b/6); %1/6th on each side, so a third in total - 45cm down to 30cm
         ReconImage_crop = ReconImage(1:(a - crop_axial), crop_lateral: (b - crop_lateral));
         Recon_abs = abs(ReconImage_crop);
         Recon_gray = mat2gray(Recon_abs,[min(Recon_abs(:)) max(Recon_abs(:))]);
         Recon_clean = zeros(size(Recon_abs));
         Recon_clean(28:end, :) = Recon_abs(28:end,:);
         Recon_clean_gray = mat2gray(Recon_clean,[min(Recon_clean(:)) max(Recon_clean(:))]);
         Recon_256 = imresize(Recon_gray, [256 256]);
         Recon_256_clean = imresize(Recon_clean_gray, [256 256]);
         imwrite(Recon_256, strcat('/wheat_processed/val/img/', saveImage));
         imwrite(Recon_256_clean, strcat('/wheat_processed/val/clean/img/', saveImage));
         
         % Extracting and Plotting Reconstructed Image - 500 m/s
         ReconImage_crop_500 = ReconImage_500(1:(a - crop_axial), crop_lateral: (b - crop_lateral));
         Recon_abs_500 = abs(ReconImage_crop_500);
         Recon_gray_500 = mat2gray(Recon_abs_500,[min(Recon_abs_500(:)) max(Recon_abs_500(:))]);
         Recon_clean_500 = zeros(size(Recon_abs_500));
         Recon_clean_500(28:end, :) = Recon_abs_500(28:end,:);
         Recon_clean_gray_500 = mat2gray(Recon_clean_500,[min(Recon_clean_500(:)) max(Recon_clean_500(:))]);
         Recon_256_500 = imresize(Recon_gray_500, [256 256]);
         Recon_256_clean_500 = imresize(Recon_clean_gray_500, [256 256]);
         imwrite(Recon_256_500, strcat('/wheat_processed/val/img_500/', saveImage));
         imwrite(Recon_256_clean_500, strcat('/wheat_processed/val/clean/img_500/', saveImage));
         
         % Extracting and Plotting Reconstructed Image - 650 m/s
         ReconImage_crop_650 = ReconImage_650(1:(a - crop_axial), crop_lateral: (b - crop_lateral));
         Recon_abs_650 = abs(ReconImage_crop_650);
         Recon_gray_650 = mat2gray(Recon_abs_650,[min(Recon_abs_650(:)) max(Recon_abs_650(:))]);
         Recon_clean_650 = zeros(size(Recon_abs_650));
         Recon_clean_650(28:end, :) = Recon_abs_650(28:end,:);
         Recon_clean_gray_650 = mat2gray(Recon_clean_650,[min(Recon_clean_650(:)) max(Recon_clean_650(:))]);
         Recon_256_650 = imresize(Recon_gray_650, [256 256]);
         Recon_256_clean_650 = imresize(Recon_clean_gray_650, [256 256]);
         imwrite(Recon_256_650, strcat('/wheat_processed/val/img_650/', saveImage));
         imwrite(Recon_256_clean_650, strcat('/wheat_processed/val/clean/img_650/', saveImage));
         
         % Extracting and Plotting Reconstructed Image - 800 m/s
         ReconImage_crop_800 = ReconImage_800(1:(a - crop_axial), crop_lateral: (b - crop_lateral));
         Recon_abs_800 = abs(ReconImage_crop_800);
         Recon_gray_800 = mat2gray(Recon_abs_800,[min(Recon_abs_800(:)) max(Recon_abs_800(:))]);
         Recon_clean_800 = zeros(size(Recon_abs_800));
         Recon_clean_800(28:end, :) = Recon_abs_800(28:end,:);
         Recon_clean_gray_800 = mat2gray(Recon_clean_800,[min(Recon_clean_800(:)) max(Recon_clean_800(:))]);
         Recon_256_800 = imresize(Recon_gray_800, [256 256]);
         Recon_256_clean_800 = imresize(Recon_clean_gray_800, [256 256]);
         imwrite(Recon_256_800, strcat('/wheat_processed/val/img_800/', saveImage));
         imwrite(Recon_256_clean_800, strcat('/wheat_processed/val/clean/img_800/', saveImage));
         
         % Combining 500, 650 and 800 to one unfocused image
         ReconImage_abs = cat(3, Recon_abs_500, Recon_abs_650, Recon_abs_800);
         Recon_gray = mat2gray(ReconImage_abs,[min(ReconImage_abs(:)) max(ReconImage_abs(:))]);
         Recon_256_uimg = imresize(Recon_gray, [256 256]);
         
         ReconImage_abs_clean = cat(3, Recon_clean_500, Recon_clean_650, Recon_clean_800);
         Recon_gray_clean = mat2gray(ReconImage_abs_clean,[min(ReconImage_abs_clean(:)) max(ReconImage_abs_clean(:))]);
         Recon_256_uimg_clean = imresize(Recon_gray_clean, [256 256]);
         
         imwrite(Recon_256_uimg, strcat('/wheat_processed/val/uimg/', saveImage));
         imwrite(Recon_256_uimg_clean, strcat('/wheat_processed/val/clean/uimg/', saveImage));
         
         % Extracting and Plotting A-Scan Data
         t_end_new = t_end*sqrt(((a - crop_axial)^2 + (b-crop_lateral)^2)/(a^2 + b^2));
         data_crop = data_down(:,1:round(t_end_new*Fs));
         data_TGC = data_crop(1:end,:).*sqrt(timeGain(1:size(data_crop,1)))';
         Ascan_gray = mat2gray(data_TGC, [-1*max(abs(data_TGC(:))) max(abs(data_TGC(:)))]); % converting to [0,1]
         Ascan_256 = imresize(Ascan_gray, [256 256]); %resampling and resizing to 256x256
         imwrite(Ascan_256', strcat('/wheat_processed/val/ascan/', saveImage));
        
            
         ctr = ctr+1;
    
end

% Test Data

ctr = 1;
for test = 9501:10000
        
         % Loading saved k-wave data
         LoadData = sprintf('/kwave_results/wheat/raw/wroot*_example%d.mat',example_vector_saved(test));
         matfile = dir(LoadData);
         load(fullfile(matfile.folder,matfile.name));
         saveImage = sprintf('%d.png',ctr);
        
         % Extracting and Plotting Reconstructed Image
         [a, b] = size(roots2); 
         crop_axial = round(a/3); %data to be removed from the bottom
         crop_lateral = round(b/6); %1/6th on each side, so a third in total - 45cm down to 30cm
         ReconImage_crop = roots2(1:(a - crop_axial), crop_lateral: (b - crop_lateral));
         Recon_abs = abs(ReconImage_crop);
         Recon_gray = mat2gray(Recon_abs,[min(Recon_abs(:)) max(Recon_abs(:))]);
         Recon_256 = round(imresize(Recon_gray, [256 256]));
         imwrite(Recon_256, strcat('/wheat_processed/test/img_bin/', saveImage));
         
         %  Extracting and Plotting corrected 8 bit SoS Map 
         [a, b] = size(SOSmap); 
         crop_axial = round(a/3); %data to be removed from the bottom
         crop_lateral = round(b/6); %1/6th on each side, so a third in total - 45cm down to 30cm
         SOSmap_crop = SOSmap(1:(a - crop_axial), crop_lateral: (b - crop_lateral));
         SOSmap_crop8 = (SOSmap_crop-340)./460;
         SOSmap_256 = imresize(SOSmap_crop8, [256 256]);  % resizing to 256x256
         imwrite(SOSmap_256, strcat('/wheat_processed/test/sos/', saveImage));


         % Extracting and Plotting Reconstructed Image
         [a, b] = size(ReconImage); 
         crop_axial = round(a/3); %data to be removed from the bottom
         crop_lateral = round(b/6); %1/6th on each side, so a third in total - 45cm down to 30cm
         ReconImage_crop = ReconImage(1:(a - crop_axial), crop_lateral: (b - crop_lateral));
         Recon_abs = abs(ReconImage_crop);
         Recon_gray = mat2gray(Recon_abs,[min(Recon_abs(:)) max(Recon_abs(:))]);
         Recon_clean = zeros(size(Recon_abs));
         Recon_clean(28:end, :) = Recon_abs(28:end,:);
         Recon_clean_gray = mat2gray(Recon_clean,[min(Recon_clean(:)) max(Recon_clean(:))]);
         Recon_256 = imresize(Recon_gray, [256 256]);
         Recon_256_clean = imresize(Recon_clean_gray, [256 256]);
         imwrite(Recon_256, strcat('/wheat_processed/test/img/', saveImage));
         imwrite(Recon_256_clean, strcat('/wheat_processed/test/clean/img/', saveImage));
         
         % Extracting and Plotting Reconstructed Image - 500 m/s
         ReconImage_crop_500 = ReconImage_500(1:(a - crop_axial), crop_lateral: (b - crop_lateral));
         Recon_abs_500 = abs(ReconImage_crop_500);
         Recon_gray_500 = mat2gray(Recon_abs_500,[min(Recon_abs_500(:)) max(Recon_abs_500(:))]);
         Recon_clean_500 = zeros(size(Recon_abs_500));
         Recon_clean_500(28:end, :) = Recon_abs_500(28:end,:);
         Recon_clean_gray_500 = mat2gray(Recon_clean_500,[min(Recon_clean_500(:)) max(Recon_clean_500(:))]);
         Recon_256_500 = imresize(Recon_gray_500, [256 256]);
         Recon_256_clean_500 = imresize(Recon_clean_gray_500, [256 256]);
         imwrite(Recon_256_500, strcat('/wheat_processed/test/img_500/', saveImage));
         imwrite(Recon_256_clean_500, strcat('/wheat_processed/test/clean/img_500/', saveImage));
         
         % Extracting and Plotting Reconstructed Image - 650 m/s
         ReconImage_crop_650 = ReconImage_650(1:(a - crop_axial), crop_lateral: (b - crop_lateral));
         Recon_abs_650 = abs(ReconImage_crop_650);
         Recon_gray_650 = mat2gray(Recon_abs_650,[min(Recon_abs_650(:)) max(Recon_abs_650(:))]);
         Recon_clean_650 = zeros(size(Recon_abs_650));
         Recon_clean_650(28:end, :) = Recon_abs_650(28:end,:);
         Recon_clean_gray_650 = mat2gray(Recon_clean_650,[min(Recon_clean_650(:)) max(Recon_clean_650(:))]);
         Recon_256_650 = imresize(Recon_gray_650, [256 256]);
         Recon_256_clean_650 = imresize(Recon_clean_gray_650, [256 256]);
         imwrite(Recon_256_650, strcat('/wheat_processed/test/img_650/', saveImage));
         imwrite(Recon_256_clean_650, strcat('/wheat_processed/test/clean/img_650/', saveImage));
         
         % Extracting and Plotting Reconstructed Image - 800 m/s
         ReconImage_crop_800 = ReconImage_800(1:(a - crop_axial), crop_lateral: (b - crop_lateral));
         Recon_abs_800 = abs(ReconImage_crop_800);
         Recon_gray_800 = mat2gray(Recon_abs_800,[min(Recon_abs_800(:)) max(Recon_abs_800(:))]);
         Recon_clean_800 = zeros(size(Recon_abs_800));
         Recon_clean_800(28:end, :) = Recon_abs_800(28:end,:);
         Recon_clean_gray_800 = mat2gray(Recon_clean_800,[min(Recon_clean_800(:)) max(Recon_clean_800(:))]);
         Recon_256_800 = imresize(Recon_gray_800, [256 256]);
         Recon_256_clean_800 = imresize(Recon_clean_gray_800, [256 256]);
         imwrite(Recon_256_800, strcat('/wheat_processed/test/img_800/', saveImage));
         imwrite(Recon_256_clean_800, strcat('/wheat_processed/test/clean/img_800/', saveImage));
         
         % Combining 500, 650 and 800 to one unfocused image
         ReconImage_abs = cat(3, Recon_abs_500, Recon_abs_650, Recon_abs_800);
         Recon_gray = mat2gray(ReconImage_abs,[min(ReconImage_abs(:)) max(ReconImage_abs(:))]);
         Recon_256_uimg = imresize(Recon_gray, [256 256]);
         
         ReconImage_abs_clean = cat(3, Recon_clean_500, Recon_clean_650, Recon_clean_800);
         Recon_gray_clean = mat2gray(ReconImage_abs_clean,[min(ReconImage_abs_clean(:)) max(ReconImage_abs_clean(:))]);
         Recon_256_uimg_clean = imresize(Recon_gray_clean, [256 256]);
         
         imwrite(Recon_256_uimg, strcat('/wheat_processed/test/uimg/', saveImage));
         imwrite(Recon_256_uimg_clean, strcat('/wheat_processed/test/clean/uimg/', saveImage));
         
         % Extracting and Plotting A-Scan Data
         t_end_new = t_end*sqrt(((a - crop_axial)^2 + (b-crop_lateral)^2)/(a^2 + b^2));
         data_crop = data_down(:,1:round(t_end_new*Fs));
         data_TGC = data_crop(1:end,:).*sqrt(timeGain(1:size(data_crop,1)))';
         Ascan_gray = mat2gray(data_TGC, [-1*max(abs(data_TGC(:))) max(abs(data_TGC(:)))]); % converting to [0,1]
         Ascan_256 = imresize(Ascan_gray, [256 256]); %resampling and resizing to 256x256
         imwrite(Ascan_256', strcat('/wheat_processed/test/ascan/', saveImage));
       
         
         
         ctr = ctr+1;
    
end

%% Be Careful with this!
