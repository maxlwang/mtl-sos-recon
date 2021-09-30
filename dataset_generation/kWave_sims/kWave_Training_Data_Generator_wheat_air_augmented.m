clear all;
clc;


%% Define k-wave grid
cair = 340;
c0_min = 500; % assuming SOS from 500-800m/s based on soil type
c0_max = 800;
pair = 1.225;
psoil = 1000; % Within the range mentioned in the Oelze Paper

points_per_wavelength = 4;
f_max = 100e3;
x_size = 0.45;
y_size = 0.45;
dx = min(cair,c0_min)/(points_per_wavelength*f_max);
Nx = 536; % Find this by hand (checkFactors)                                         
dy = min(cair,c0_min)/(points_per_wavelength*f_max);
Ny = 536; % Find this by hand (checkFactors)  
x_size = Nx*dx;
y_size = Ny*dy;
kgrid = kWaveGrid(Nx,dx,Ny,dy);


% Define CMUT Standoff
cmut_height = 0.05;
medium.sound_speed = cair*ones(Nx,Ny);
medium.density = pair*ones(Nx,Ny);

x_size_soil = Nx*dx - cmut_height;
Nx_soil = Nx - round(cmut_height/dx) + 1;
medium.sound_speed(round(cmut_height/dx):end,:) = c0_min;
medium.density(round(cmut_height/dx):end,:) = psoil;


% Define Time Varying Source
% Sampling Frequency and Duration
Fs = 5e6;
t_end = 1.3e-3;
kgrid.t_array = 0:1/Fs:t_end-1/Fs;

% Define CMUTs
Q = 5; 
source_freq = 100e3;
wc= 2*pi*source_freq;
delw= wc/Q;
H = tf([delw 0], [1 delw wc^2]);
        
                
% Excitation and Suppression Source
source_mag = 1;
no_excite = 2;
        
time_excite = 0:1/Fs:no_excite/source_freq;
time_suppress = 0:1/Fs:1/source_freq;
signal_excite = source_mag*(sin(2*pi*source_freq*time_excite));
signal_suppress = 0.594*source_mag*(sin(2*pi*source_freq*time_suppress));
   
% Combined Source
signal_excite_suppress_mask = [signal_excite zeros(1,round(0.5*Fs/source_freq)) signal_suppress];
signal_excite_suppress = [signal_excite_suppress_mask zeros(1,length(kgrid.t_array)-length(signal_excite_suppress_mask))];
        
        
% k-wave source
source.p = signal_excite_suppress;
        
% Define Sensors
sensorMask = 1:2:Ny;
sensor.mask = zeros(Nx, Ny);
sensor.mask(1,sensorMask) = 1;   
sensor.record = {'p', 'p_final'};
        
% % Run the Simulation - % define PML outside and plotting off
input_args = {'PMLInside', false, 'DataCast', 'gpuArray-single', 'DataRecast',false,...
     'PlotSim', false, 'PlotLayout', false, 'PlotPML', false};

% Run the Simulation - 
% input_args = {'PMLInside', false, 'DataCast', 'gpuArray-single', 'DataRecast',false,...
%             'PlotSim', false, 'PlotLayout', false, 'PlotPML', false, 'RecordMovie', true, 'MovieName', 'example_movie2'};
     
CMUTlocs = [ones(1,Ny/2);1:Ny/2];  

SOSmap_500 = 500*ones(Nx,Ny);  %Uniform SOS
SOSmap_650 = 650*ones(Nx,Ny);  %Uniform SOS
SOSmap_800 = 800*ones(Nx,Ny);  %Uniform SOS
SOSmap_500(1:round(cmut_height/dx),:) = cair;
SOSmap_650(1:round(cmut_height/dx),:) = cair;
SOSmap_800(1:round(cmut_height/dx),:) = cair;
SOSmap_500 = imresize(SOSmap_500,[Nx/2,Ny/2]);
SOSmap_650 = imresize(SOSmap_650,[Nx/2,Ny/2]);
SOSmap_800 = imresize(SOSmap_800,[Nx/2,Ny/2]);

TOAmaps_500 = FastMarchingMethod(SOSmap_500,CMUTlocs,2*dx);
TOAmaps_650 = FastMarchingMethod(SOSmap_650,CMUTlocs,2*dx);
TOAmaps_800 = FastMarchingMethod(SOSmap_800,CMUTlocs,2*dx);
        
%% Generating 5000 images, with 5 SOS for each of the 1000 roots
number_of_roots = 1000;
number_SOS_per_root = 5;
for sim_num = 1:number_of_roots
    for n = 1:number_SOS_per_root
        
        % Read in roots structure (pre-generate all of these) - Max
        rootFile = sprintf('wroot%d.png',sim_num);
        roots = imread(strcat('', rootFile));
        roots = abs(rescale(roots));
        
        roots2 = zeros(Nx, Ny);
        roots2(1:round(cmut_height/dx),:) = 0;
        roots2(round(cmut_height/dx):end, :) = round(imresize(roots, [Nx_soil, Ny]));
        
        % Define the roots as sources (source.p_mask) + scattered targets
        source.p_mask = roots2;

        % Generate Speed of Sound Distribution
        medium.sound_speed(round(cmut_height/dx):end,:) = RandomSOSmapGenerator([c0_min,c0_max],[Nx_soil,Ny],x_size_soil, y_size);
              
        sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
        signals = sensor_data.p;
        
        % Convolve with CMUT Response 
        data = zeros(size(signals));
        for i = 1:min(size(signals))
            data(i,:) = lsim(H, gather(double(signals(i,:))), kgrid.t_array);
        end
        
        SNR_target = 20;
        SNR = 0;
        sig = 0.34e-3; % SNR=inf,sig=0 SNR=10,sig=0.3e-5 SNR=5,sig=0.54e-5 SNR=-5,sig=1.7e-5 SNR=-15,sig=5.4e-5
        ctr = 0;
        
        while((abs(SNR_target - SNR) > 0.02) && ctr < 10)
            ctr = ctr + 1;
            noise = sig*randn(size(data));
            
            % Matched Filter Data
            Fs2 = 5e6;
            matched_filter = lsim(H,signal_excite_suppress,kgrid.t_array);
            data_down = zeros(size(data,1),size(data,2)*Fs2/Fs);
            noise_down = zeros(size(data,1),size(data,2)*Fs2/Fs);
            time2 = 0:1/Fs2:t_end-1/Fs2;
            for i = 1:size(data,1)
                filtered = xcorr(data(i,:),matched_filter);
                noise_filtered = xcorr(noise(i,:),matched_filter);
                data_filtered = filtered(round(length(filtered)/2):end);
                noise_filtered = noise_filtered(round(length(noise_filtered)/2):end);
                data_down(i,:) = resample(data_filtered, Fs2, Fs);
                noise_down(i,:) = resample(noise_filtered, Fs2, Fs);
            end
            SNR = snr(data_down,noise_down);
            sig = sig*10^(-(SNR_target - SNR)/20);
        end
        
        data_down = data_down+noise_down;
        
        %% Reconstruct the Ground Truth Image 
        SOSmap = medium.sound_speed;
        SOSmap = imresize(SOSmap,[Nx/2,Ny/2]);
        
        TOAmaps = FastMarchingMethod(SOSmap,CMUTlocs,2*dx);
      
        
        ReconImage = BackProjj_2D_SoundMap(hilbert(data_down'),Fs2,TOAmaps);
        ReconImage_500 = BackProjj_2D_SoundMap(hilbert(data_down'),Fs2,TOAmaps_500);
        ReconImage_650 = BackProjj_2D_SoundMap(hilbert(data_down'),Fs2,TOAmaps_650);
        ReconImage_800 = BackProjj_2D_SoundMap(hilbert(data_down'),Fs2,TOAmaps_800);


        % Save SoSmap, Data, Ground Truth Image 
        saveFile = sprintf('wroot%d_example%d.mat',sim_num,(sim_num-1)*number_SOS_per_root+n);
        save(strcat('/kwave_results/wheat/raw/', saveFile), 'ReconImage_500', 'ReconImage_650', 'ReconImage_800', 'ReconImage', 'SOSmap','data', 'Fs', 'Fs2', 'roots2', 'data_down')
     
        saveImage = sprintf('recon_wroot%d_example%d.png',sim_num,(sim_num-1)*number_SOS_per_root+n);
        imwrite(ind2rgb(im2uint8(mat2gray(abs(ReconImage))), hot), strcat('/kwave_results/wheat/img/', saveImage));
        
        saveImage = sprintf('recon_wroot%d_example%d.png',sim_num,(sim_num-1)*number_SOS_per_root+n);
        imwrite(ind2rgb(im2uint8(mat2gray(abs(ReconImage_500))), hot), strcat('/kwave_results/wheat/uimg_500/', saveImage));
        
        saveImage = sprintf('recon_wroot%d_example%d.png',sim_num,(sim_num-1)*number_SOS_per_root+n);
        imwrite(ind2rgb(im2uint8(mat2gray(abs(ReconImage_650))), hot), strcat('/kwave_results/wheat/uimg_650/', saveImage));
        
        saveImage = sprintf('recon_wroot%d_example%d.png',sim_num,(sim_num-1)*number_SOS_per_root+n);
        imwrite(ind2rgb(im2uint8(mat2gray(abs(ReconImage_800))), hot), strcat('/kwave_results/wheat/uimg_800/', saveImage));
        

    end

end

%% Generating 5000 images, with 5 roots for each of the 1000 SoS distributions
number_of_SOS = 1000;
number_roots_per_SOS = 5;
for sim_num = 1:number_of_SOS
    
    % Generate Speed of Sound Distribution
    medium.sound_speed(round(cmut_height/dx):end,:) = RandomSOSmapGenerator([c0_min,c0_max],[Nx_soil,Ny],x_size_soil, y_size);
    
    for n = 1:number_roots_per_SOS

        root_num = randi([1,2000],1,1); 
        
        % Read in roots structure (pre-generate all of these)
        rootFile = sprintf('wroot%d.png',root_num);
        roots = imread(strcat('/binwroot/', rootFile));
        roots = abs(rescale(roots));
        
        roots2 = zeros(Nx, Ny);
        roots2(1:round(cmut_height/dx),:) = 0;
        roots2(round(cmut_height/dx):end, :) = round(imresize(roots, [Nx_soil, Ny]));
        
        % Define the roots as sources (source.p_mask) + scattered targets
        source.p_mask = roots2;
              
        sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
        signals = sensor_data.p;
        
        % Convolve with CMUT Response 
        data = zeros(size(signals));
        for i = 1:min(size(signals))
            data(i,:) = lsim(H, gather(double(signals(i,:))), kgrid.t_array);
        end
        
        SNR_target = 20;
        SNR = 0;
        sig = 0.34e-3; % SNR=inf,sig=0 SNR=10,sig=0.3e-5 SNR=5,sig=0.54e-5 SNR=-5,sig=1.7e-5 SNR=-15,sig=5.4e-5
        ctr = 0;
        
        while((abs(SNR_target - SNR) > 0.02) && ctr < 10)
            ctr = ctr + 1;
            noise = sig*randn(size(data));
            
            % Matched Filter Data
            Fs2 = 5e6;
            matched_filter = lsim(H,signal_excite_suppress,kgrid.t_array);
            data_down = zeros(size(data,1),size(data,2)*Fs2/Fs);
            noise_down = zeros(size(data,1),size(data,2)*Fs2/Fs);
            time2 = 0:1/Fs2:t_end-1/Fs2;
            for i = 1:size(data,1)
                filtered = xcorr(data(i,:),matched_filter);
                noise_filtered = xcorr(noise(i,:),matched_filter);
                data_filtered = filtered(round(length(filtered)/2):end);
                noise_filtered = noise_filtered(round(length(noise_filtered)/2):end);
                data_down(i,:) = resample(data_filtered, Fs2, Fs);
                %data_down(i,:) = interp1(kgrid.t_array, data_filtered, time2); 
                noise_down(i,:) = resample(noise_filtered, Fs2, Fs);
            end
            SNR = snr(data_down,noise_down);
            sig = sig*10^(-(SNR_target - SNR)/20);
        end
        
        data_down = data_down+noise_down;
        
        %% Reconstruct the Ground Truth Image 
                
        SOSmap = medium.sound_speed;
        SOSmap = imresize(SOSmap,[Nx/2,Ny/2]);
       
        
        TOAmaps = FastMarchingMethod(SOSmap,CMUTlocs,2*dx);
      
        
        ReconImage = BackProjj_2D_SoundMap(hilbert(data_down'),Fs2,TOAmaps);
        ReconImage_500 = BackProjj_2D_SoundMap(hilbert(data_down'),Fs2,TOAmaps_500);
        ReconImage_650 = BackProjj_2D_SoundMap(hilbert(data_down'),Fs2,TOAmaps_650);
        ReconImage_800 = BackProjj_2D_SoundMap(hilbert(data_down'),Fs2,TOAmaps_800);


        % Save SoSmap, Data, Ground Truth Image 
        saveFile = sprintf('wroot%d_example%d.mat',root_num,(sim_num-1)*number_roots_per_SOS+n+ number_of_roots*number_SOS_per_root);
        save(strcat('/kwave_results/wheat/raw/', saveFile), 'ReconImage_500', 'ReconImage_650', 'ReconImage_800', 'ReconImage', 'SOSmap','data', 'Fs', 'Fs2', 'roots2', 'data_down')
     
        saveImage = sprintf('recon_wroot%d_example%d.png',root_num,(sim_num-1)*number_roots_per_SOS+n+number_of_roots*number_SOS_per_root);
        imwrite(ind2rgb(im2uint8(mat2gray(abs(ReconImage))), hot), strcat('/kwave_results/wheat/img/', saveImage));
        
        saveImage = sprintf('recon_wroot%d_example%d.png',root_num,(sim_num-1)*number_roots_per_SOS+n+number_of_roots*number_SOS_per_root);
        imwrite(ind2rgb(im2uint8(mat2gray(abs(ReconImage_500))), hot), strcat('/kwave_results/wheat/uimg_500/', saveImage));
        
        saveImage = sprintf('recon_wroot%d_example%d.png',root_num,(sim_num-1)*number_roots_per_SOS+n+number_of_roots*number_SOS_per_root);
        imwrite(ind2rgb(im2uint8(mat2gray(abs(ReconImage_650))), hot), strcat('/kwave_results/wheat/uimg_650/', saveImage));
        
        saveImage = sprintf('recon_wroot%d_example%d.png',root_num,(sim_num-1)*number_roots_per_SOS+n+number_of_roots*number_SOS_per_root);
        imwrite(ind2rgb(im2uint8(mat2gray(abs(ReconImage_800))), hot), strcat('/kwave_results/wheat/uimg_800/', saveImage));
        
       
    end

end

