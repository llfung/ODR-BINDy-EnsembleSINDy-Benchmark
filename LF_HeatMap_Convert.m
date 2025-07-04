%% Lorenz
load('Lorenz_simOutEFinal.mat','successE','modelErrorE','tEndL','epsL'); % With Golay

% tEndL=tEndL(2:2:end);
success      = mean(successE(:,:,:),3)'; % Bragging SINDy
modelError   = mean(modelErrorE(:,:,:),3)'; % Bragging SINDy

save("Lorenz_E-SINDy.mat",'success','modelError','tEndL','epsL');
%% Rossler
load('Rossler_simOutEFinal.mat','successE','modelErrorE','tEndL','epsL'); % With Golay

success      = mean(successE(:,:,:),3)'; % Bragging SINDy
modelError   = mean(modelErrorE(:,:,:),3)'; % Bragging SINDy

save("Rossler_E-SINDy.mat",'success','modelError','tEndL','epsL');
%% Van Der Pol
load('VanDerPol_simOutEFinal.mat','successE','modelErrorE','tEndL','epsL'); % With Golay

success      = mean(successE(:,:,:),3)'; % Bragging SINDy
modelError   = mean(modelErrorE(:,:,:),3)'; % Bragging SINDy

save("VanDerPol_E-SINDy.mat",'success','modelError','tEndL','epsL');