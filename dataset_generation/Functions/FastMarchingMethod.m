
% Requires download of msfm2d function from: 
    % Dirk-Jan Kroon (2021). Accurate Fast Marching (https://www.mathworks.com/matlabcentral/fileexchange/24531-accurate-fast-marching), MATLAB Central File Exchange.

function TOA = FastMarchingMethod(C,receiverLocations,dr)

    Nrec = size(receiverLocations,2);
    TOA = zeros(size(C,1),size(C,2),Nrec);
    
    for i = 1:Nrec
        
        TOA(:,:,i) = dr*msfm2d(C, receiverLocations(:,i), true, true);
    
    end
    
end