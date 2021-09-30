function Im = BackProjj_2D_SoundMap(DataArrayFromCMUT,Fs,T)


    % Time-gain
    dt = 1/Fs;
    t_end = size(DataArrayFromCMUT,1)*dt;
    t = 0:dt:t_end-dt;
    timeGain = t./t_end;
    DataArrayFromCMUT = DataArrayFromCMUT(1:end,:).*sqrt(timeGain(1:size(DataArrayFromCMUT,1)))';

    % CMUT Parameters
    numScans = min(size(DataArrayFromCMUT));
    
    tic()
    Im = zeros(size(T,1),size(T,2));
    for sn = 1:numScans
        for nX = 1:size(T,2)
            for nZ = 1:size(T,1)
                
                indT = round(T(nZ,nX,sn)*Fs);
                if indT <= 0 || indT>size(DataArrayFromCMUT,1)
                    continue;
                end
             
                Im(nZ,nX) = Im(nZ,nX) + DataArrayFromCMUT(indT,sn);
                
            end
        end
    end
    toc()
    
end