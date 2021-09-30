function SOSmap = RandomSOSmapGenerator(SOS,N,sizeX,sizeY)

    % Spatial Wavelengths
    lambda = 0.5*sizeX:0.005:1.3*sizeX;
    k = 2*pi./lambda;
    theta = 0:pi/10:2*pi;

    % 2-D Distribution
    [X,Y] = meshgrid(-sizeX/2:sizeX/N(1):sizeX/2,-sizeY/2:sizeY/N(2):sizeY/2);

    H = zeros(size(X));
    for m = 1:length(k)
        for n = 1:length(theta)
            H = H + cos(k(m)*X*cos(theta(n))+k(m)*Y*sin(theta(n))+2*pi*rand(1));
        end
    end

    avgSOS = (SOS(1) + SOS(2))/2;
    lowSOS = SOS(1) + (avgSOS - SOS(1))*rand(1);
    hiSOS = avgSOS + (SOS(2) - avgSOS)*rand(1);
    
    SOSmap = rescale(H,lowSOS,hiSOS);
    SOSmap = SOSmap';
    SOSmap = SOSmap(1:N(1),1:N(2));
    
end
