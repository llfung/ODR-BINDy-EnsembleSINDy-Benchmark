%%%%%%%%%%%%%%%%%%%
% 
% run simulations over noise level and data length for heatmap plot
%
%

clear all
% close all
% clc

addpath(genpath('colormaps'))

warning('off','MATLAB:rankDeficientMatrix');
warning('off','MATLAB:nearlySingularMatrix');
%% sweep over a set of noise levels and data length to generate heatmap plots
% noise level
epsL = 0.025:0.025:0.4;

% simulation time
tEndL = 10.0:2.0:30.0;

% at each noise level and simulation time, nTest different instantiations of noise are run (model errors and success rate are then averaged for plotting)
nTest1 = 128; % generate models nTest1 times for SINDy
nTest2 = 128; % generate models nTest times for ensemble SINDy


%% hyperparameters
% SINDy sparsifying hyperparameters
lambda = 0.05;

% ensemble hyperparameters
% data ensembling
nEnsembles = 500; % number of bootstraps (SINDy models using sampled data) in ensemble
ensembleT = 0.65; % Threshold model coefficient inclusion probability: set ensemble SINDy model coefficient to zero if inclusion probability is below ensembleT

% library
nEnsemble1P = 0.9; % percentage of full library that is sampled without replacement for library bagging
nEnsemble2 = 100; % number of bootstraps (SINDy models using sampled library terms) in ensemble
ensT = 0.4; % Threshold library term inclusion probabilty: cut library entries that occur less than ensT

% double bagging
nEnsemblesDD = 100; % number of models in ensemble for data bagging after library bagging

%% common parameters, true Rossler system, signal power for noise calculation

% generate synthetic Rossler system data
ode_params.a = 0.2; 
ode_params.b = 0.2; 
ode_params.c = 5.7; 
x0 = [-6 5 0]';
n = length(x0); 

% set common params
polys = 0:2;
trigs = [];
common_params = {polys,trigs};
gamma = 0;
tol_ode = 1e-10;         % set tolerance (abs and rel) of ode45
options = odeset('RelTol',tol_ode,'AbsTol',tol_ode*ones(1,length(x0)));

% time step
dt = 0.05;

% get true Rossler system for comparison
true_nz_weights = zeros(10 ,3);
true_nz_weights(1,:) = [      0 ,       0 , ode_params.b ];
true_nz_weights(2,:) = [      0 ,       1 ,       0 ]; % y(1)
true_nz_weights(3,:) = [     -1 , ode_params.a ,       0 ]; % y(2)
true_nz_weights(4,:) = [     -1 ,       0 , -ode_params.c ]; % y(3)
true_nz_weights(6,:) = [      0 ,       0 ,       1 ]; % y(1) * y(3)

% signal power for noise calculation
% [~,~,x10,~] = lorenz(x0,dtL(1):dtL(1):10,tol_ode,ode_params);
[~,x10]=ode45(@(t,x) rossler(t,x,ode_params),dt:dt:30,x0,options);

signal_power = std(x10(:));


%% general parameters

% smooth data using golay filter 
sgolayON = 1;

runSim = 1; % run sim or load data

if runSim
    
% choose which method(s) to run. otherwise results are loaded.
% the final heatmap plot shows SINDy (runS), baggin and bragging (runE), and library bagging (runEL and runDoubleBag)
runS = 1; % run SINDy and w-SINDy
runE = 1; % run Ensemble on data
runEL = 0; % run Ensemble on library
runJK = 0; % run jackknife sampling
runDoubleBag = 0; % run double bagging: first library then data bagging
runWR = 0; % bagging and bragging without replacement

saveTrue = 1;


%% run SINDY

if runS
    
    nWrongTermsS = zeros(length(epsL),length(tEndL),nTest1);
    modelErrorS = zeros(length(epsL),length(tEndL),nTest1);
    successS = zeros(length(epsL),length(tEndL),nTest1);

    for ieps = 1:length(epsL)
        noise_ratio = epsL(ieps);

        for idt = 1:length(tEndL)
            tEnd = tEndL(idt);

            tspan = dt:dt:tEnd;

            [t,x]=ode45(@(t,x) rossler(t,x,ode_params),tspan,x0,options);


            % set rnd number for reproduction
            rng(1,'twister')

            for ii = 1:nTest1

                % add noise
                sigma = noise_ratio*signal_power;
                noise = normrnd(0,sigma,size(x));
                xobs = x + noise;

                % smooth data
                if sgolayON 
                    order = 3;
                    framelen = 5;
                    xobs = sgolayfilt(xobs,order,framelen);
                end

                %% recover dynamics
                Theta_0 = build_theta(xobs,common_params);

                %% SINDy 
                % sindy with central difference differentiation
                sindy = sindy_cd(xobs,Theta_0,n,lambda,gamma,dt);


                %% store outputs
                nWrongTermsS(ieps,idt,ii) = sum(sum(abs((true_nz_weights~=0) - (sindy~=0))));
                modelErrorS(ieps,idt,ii) = norm(sindy-true_nz_weights,"fro")/norm(true_nz_weights,"fro");
                successS(ieps,idt,ii) = norm((true_nz_weights~=0) - (sindy~=0))==0;

            end
        end
        % if saveTrue
        %     save(sprintf('simOutS%.0f',ieps))
        % end
    end

    if saveTrue
        save('Rossler_simOutSFinal')
    end
end



%% ENSEMBLES SINDY

if runE || runEL

if runE
    nWrongTermsE = zeros(length(epsL),length(tEndL),nTest2);
    nWrongTermsE2 = zeros(length(epsL),length(tEndL),nTest2);
    modelErrorE = zeros(length(epsL),length(tEndL),nTest2);
    modelErrorE2 = zeros(length(epsL),length(tEndL),nTest2);
    successE = zeros(length(epsL),length(tEndL),nTest2);
    successE2 = zeros(length(epsL),length(tEndL),nTest2);

    nWrongTermsEoos = zeros(length(epsL),length(tEndL),nTest2);
    modelErrorEoos = zeros(length(epsL),length(tEndL),nTest2);
    successEoos = zeros(length(epsL),length(tEndL),nTest2);
    nWrongTermsEoos2 = zeros(length(epsL),length(tEndL),nTest2);
    modelErrorEoos2 = zeros(length(epsL),length(tEndL),nTest2);
    successEoos2 = zeros(length(epsL),length(tEndL),nTest2);
    
    % ensembling without replacement
    nWrongTermsWRE = zeros(length(epsL),length(tEndL),nTest2);
    nWrongTermsWRE2 = zeros(length(epsL),length(tEndL),nTest2);
    modelErrorWRE = zeros(length(epsL),length(tEndL),nTest2);
    modelErrorWRE2 = zeros(length(epsL),length(tEndL),nTest2);
    successWRE = zeros(length(epsL),length(tEndL),nTest2);
    successWRE2 = zeros(length(epsL),length(tEndL),nTest2);
end

if runEL
    nWrongTermsDE = zeros(length(epsL),length(tEndL),nTest2);
    nWrongTermsDDE = zeros(length(epsL),length(tEndL),nTest2);
    nWrongTermsDDE2 = zeros(length(epsL),length(tEndL),nTest2);
    
    modelErrorDE = zeros(length(epsL),length(tEndL),nTest2);
    modelErrorDDE = zeros(length(epsL),length(tEndL),nTest2);
    modelErrorDDE2 = zeros(length(epsL),length(tEndL),nTest2);
    
    successDE = zeros(length(epsL),length(tEndL),nTest2);
    successDDE = zeros(length(epsL),length(tEndL),nTest2);
    successDDE2 = zeros(length(epsL),length(tEndL),nTest2);
    
    % Jackknife Resampling
    nWrongTermsDEJK = zeros(length(epsL),length(tEndL),nTest2);
    modelErrorDEJK = zeros(length(epsL),length(tEndL),nTest2);
    successDEJK = zeros(length(epsL),length(tEndL),nTest2);
end

for ieps = 1:length(epsL)
    noise_ratio = epsL(ieps);

    for idt = 1:length(tEndL)
        tEnd = tEndL(idt);
        tspan = dt:dt:tEnd;

        [t,x]=ode45(@(t,x) rossler(t,x,ode_params),tspan,x0,options);

        
        
        rng(1,'twister')
        
        parfor ii = 1:nTest2
            warning('off','MATLAB:rankDeficientMatrix');
            warning('off','MATLAB:nearlySingularMatrix');
            % add noise
            sigma = noise_ratio*signal_power;        
            noise = normrnd(0,sigma,size(x));
            xobs = x + noise;

            % smooth
            if sgolayON 
                order = 3;
                framelen = 5;
                xobs = sgolayfilt(xobs,order,framelen);
            end

            %% recover dynamics
            Theta_0 = build_theta(xobs,common_params);

            %% calculate derivatives
            % finite difference differentiation
            dxobs_0 = zeros(size(x));
            dxobs_0(1,:)=(-11/6*xobs(1,:) + 3*xobs(2,:) -3/2*xobs(3,:) + xobs(4,:)/3)/dt;
            dxobs_0(2:size(xobs,1)-1,:) = (xobs(3:end,:)-xobs(1:end-2,:))/(2*dt);
            dxobs_0(size(xobs,1),:) = (11/6*xobs(end,:) - 3*xobs(end-1,:) + 3/2*xobs(end-2,:) - xobs(end-3,:)/3)/dt;
            
            if runE

                %% SINDy ensemble

%                 bootstat = bootstrp(nEnsembles,@(Theta,dx)sparsifyDynamics(Theta,dx,lambda,n,gamma),Theta_0,dxobs_0);
                [bootstat,bootstatn] = bootstrp(nEnsembles,@(Theta,dx)sparsifyDynamics(Theta,dx,lambda,n,gamma),Theta_0,dxobs_0); 
                XiE = zeros(size(Theta_0,2),n,nEnsembles);
                XiEnz = zeros(size(Theta_0,2),n,nEnsembles);
                for iE = 1:nEnsembles
                    XiE(:,:,iE) = reshape(bootstat(iE,:),size(Theta_0,2),n);
                    XiEnz(:,:,iE) = XiE(:,:,iE)~=0;
                end

                % only consider ensemble members with small out of sample error
                OOSeps = zeros(1,nEnsembles);
                for jj = 1:nEnsembles
                    XX = [1:size(Theta_0,1), bootstatn(:,jj)'];
                    nUnique = histc(XX, unique(XX));
                    uniqueVals = find(nUnique == 1);

                    OOSeps(jj) = sum(abs(1-mean((Theta_0(uniqueVals,:)*XiE(:,:,jj))./dxobs_0(uniqueVals,:))));
                end
                [~,OOSsmall] = mink(OOSeps,round(0.1*nEnsembles)); % choose 10% best models

                
                % Thresholded bootstrap aggregating (bagging, from bootstrap aggregating)
                XiEnzM = mean(XiEnz,3); % mean of non-zero values in ensemble
                XiEnzM(XiEnzM<ensembleT) = 0; % threshold: set all parameters that have an inclusion probability below threshold to zero

                Xi = mean(XiE,3);
                XiMedian = median(XiE,3);
                XiOOS = mean(XiE(:,:,OOSsmall),3);
                XiOOSmed = median(XiE(:,:,OOSsmall),3);

                Xi(XiEnzM==0)=0; 
                XiMedian(XiEnzM==0)=0; 
                XiOOS(XiEnzM==0)=0; 
                XiOOSmed(XiEnzM==0)=0; 
                
                
                %% SINDy ensemble without replacement
                % randomly sample data without replacement
                if runWR
                    nEnsembleWR1 = round(0.5*size(Theta_0,1));
                    nEnsembleWR2 = nEnsembles;
                    XiWRE = zeros(size(Theta_0,2),n,nEnsembleWR2);
                    XiWREnz = zeros(size(Theta_0,2),n,nEnsembleWR2);
                    libOutBSWR = zeros(nEnsembleWR1,nEnsembleWR2);
                    for iii = 1:nEnsembleWR2
                        rs = RandStream('mlfg6331_64','Seed',iii); 
                        libOutBSWR(:,iii) = datasample(rs,1:size(Theta_0,1),nEnsembleWR1,'Replace',false)';
                        XiWRE(:,:,iii) = sparsifyDynamics(Theta_0(libOutBSWR(:,iii),:),dxobs_0(libOutBSWR(:,iii),:),lambda,n,gamma);
                        XiWREnz(:,:,iii) = XiWRE(:,:,iii)~=0;
                    end

                    % Thresholded bootstrap aggregating (bagging, from bootstrap aggregating)
                    XiWREnzM = mean(XiWREnz,3); % mean of non-zero values in ensemble
                    XiWREnzM(XiWREnzM<ensembleT) = 0; % threshold: set all parameters that have an inclusion probability below threshold to zero

                    XiWR = mean(XiWRE,3);
                    XiWRMedian = median(XiWRE,3);

                    XiWR(XiWREnzM==0)=0; 
                    XiWRMedian(XiWREnzM==0)=0; 
                end
            end


            %% double bagging SINDy
            if runEL
            
                %% Bagging SINDy library
                % randomly sample library terms without replacement and throw away terms
                % with low inclusion probability
                nEnsemble1 = round(nEnsemble1P*size(Theta_0,2));
                mOutBS = zeros(nEnsemble1,n,nEnsemble2);
                libOutBS = zeros(nEnsemble1,nEnsemble2);
                for iii = 1:nEnsemble2
                    rs = RandStream('mlfg6331_64','Seed',iii); 
                    libOutBS(:,iii) = datasample(rs,1:size(Theta_0,2),nEnsemble1,'Replace',false)';
                    mOutBS(:,:,iii) = sparsifyDynamics(Theta_0(:,libOutBS(:,iii)),dxobs_0,lambda,n,gamma);
                end

                inclProbBS = zeros(size(Theta_0,2),n);
                for iii = 1:nEnsemble2
                    for jjj = 1:n
                        for kkk = 1:nEnsemble1
                            if mOutBS(kkk,jjj,iii) ~= 0
                                inclProbBS(libOutBS(kkk,iii),jjj) = inclProbBS(libOutBS(kkk,iii),jjj) + 1;
                            end
                        end
                    end
                end
                inclProbBS = inclProbBS/nEnsemble2*size(Theta_0,2)/nEnsemble1;

                XiD = zeros(size(Theta_0,2),n);
                for iii = 1:n
                    libEntry = inclProbBS(:,iii)>ensT;
                    XiBias = sparsifyDynamics(Theta_0(:,libEntry),dxobs_0(:,iii),lambda,1,gamma);
                    XiD(libEntry,iii) = XiBias;
                end

                %% Jacknife sampling bagging SINDy library
                % For a sample with n points, the jackknife computes sample statistics on n separate samples of size n-1. 
                % Each sample is the original data with a single observation omitted.
                if runJK
                    nEnsembleJK1 = size(Theta_0,2)-1;
                    nEnsembleJK2 = size(Theta_0,2);
                    mOutBSJK = zeros(nEnsembleJK1,n,nEnsembleJK2);
                    libOutBSJK = zeros(nEnsembleJK1,nEnsembleJK2);
                    libOutBSJKiii = 1:nEnsembleJK2;
                    for iii = 1:nEnsembleJK2             
                        libOutBSJK(:,iii) = libOutBSJKiii(libOutBSJKiii~=iii)';
                        mOutBSJK(:,:,iii) = sparsifyDynamics(Theta_0(:,libOutBSJK(:,iii)),dxobs_0,lambda,n,gamma);
                    end

                    inclProbBSJK = zeros(size(Theta_0,2),n);
                    for iii = 1:nEnsembleJK2
                        for jjj = 1:n
                            for kkk = 1:nEnsemble1
                                if mOutBSJK(kkk,jjj,iii) ~= 0
                                    inclProbBSJK(libOutBSJK(kkk,iii),jjj) = inclProbBSJK(libOutBSJK(kkk,iii),jjj) + 1;
                                end
                            end
                        end
                    end
                    inclProbBSJK = inclProbBSJK/nEnsembleJK2*size(Theta_0,2)/nEnsembleJK1;

                    XiDJK = zeros(size(Theta_0,2),n);
                    for iii = 1:n
                        libEntry = inclProbBSJK(:,iii)>ensT;
                        XiBias = sparsifyDynamics(Theta_0(:,libEntry),dxobs_0(:,iii),lambda,1,gamma);
                        XiDJK(libEntry,iii) = XiBias;
                    end
                end

                
                %% Double bagging SINDy 
                % randomly sample library terms without replacement and throw away terms
                % with low inclusion probability
                % then on smaller library, do bagging
   
                if runDoubleBag
                    XiDB = zeros(size(Theta_0,2),n);
                    XiDBmed = zeros(size(Theta_0,2),n);
                    for iii = 1:n
                        libEntry = inclProbBS(:,iii)>ensT;

                        bootstatDD = bootstrp(nEnsemblesDD,@(Theta,dx)sparsifyDynamics(Theta,dx,lambda,1,gamma),Theta_0(:,libEntry),dxobs_0(:,iii)); 

                        XiDBe = [];
                        XiDBnz = [];
                        for iE = 1:nEnsemblesDD
                            XiDBe(:,iE) = reshape(bootstatDD(iE,:),size(Theta_0(:,libEntry),2),1);
                            XiDBnz(:,iE) = XiDBe(:,iE)~=0;
                        end

                        % Thresholded bootstrap aggregating (bagging, from bootstrap aggregating)
                        XiDBnzM = mean(XiDBnz,2); % mean of non-zero values in ensemble
                        XiDBnzM(XiDBnzM<ensembleT) = 0; % threshold: set all parameters that have an inclusion probability below threshold to zero

                        XiDBmean = mean(XiDBe,2);
                        XiDBmedian = median(XiDBe,2);

                        XiDBmean(XiDBnzM==0)=0; 
                        XiDBmedian(XiDBnzM==0)=0; 

                        XiDB(libEntry,iii) = XiDBmean;
                        XiDBmed(libEntry,iii) = XiDBmedian;

                    end
                end
            end
            
            %% store outputs

            if runE
                
                nWrongTermsE(ieps,idt,ii) = sum(sum(abs((true_nz_weights~=0) - (XiMedian~=0))));
                nWrongTermsE2(ieps,idt,ii) = sum(sum(abs((true_nz_weights~=0) - (Xi~=0))));
                modelErrorE(ieps,idt,ii) = norm(XiMedian-true_nz_weights,"fro")/norm(true_nz_weights,"fro");
                modelErrorE2(ieps,idt,ii) = norm(Xi-true_nz_weights,"fro")/norm(true_nz_weights,"fro");
                successE(ieps,idt,ii) = norm((true_nz_weights~=0) - (XiMedian~=0))==0;
                successE2(ieps,idt,ii) = norm((true_nz_weights~=0) - (Xi~=0))==0;
                
                nWrongTermsEoos(ieps,idt,ii) = sum(sum(abs((true_nz_weights~=0) - (XiOOS~=0))));
                modelErrorEoos(ieps,idt,ii) = norm(XiOOS-true_nz_weights,"fro")/norm(true_nz_weights,"fro");
                successEoos(ieps,idt,ii) = norm((true_nz_weights~=0) - (XiOOS~=0))==0;
                nWrongTermsEoos2(ieps,idt,ii) = sum(sum(abs((true_nz_weights~=0) - (XiOOSmed~=0))));
                modelErrorEoos2(ieps,idt,ii) = norm(XiOOSmed-true_nz_weights,"fro")/norm(true_nz_weights,"fro");
                successEoos2(ieps,idt,ii) = norm((true_nz_weights~=0) - (XiOOSmed~=0))==0;
                
                if runWR
                    nWrongTermsWRE(ieps,idt,ii) = sum(sum(abs((true_nz_weights~=0) - (XiWRMedian~=0))));
                    nWrongTermsWRE2(ieps,idt,ii) = sum(sum(abs((true_nz_weights~=0) - (XiWR~=0))));
                    modelErrorWRE(ieps,idt,ii) = norm(XiWRMedian-true_nz_weights,"fro")/norm(true_nz_weights,"fro");
                    modelErrorWRE2(ieps,idt,ii) = norm(XiWR-true_nz_weights,"fro")/norm(true_nz_weights,"fro");
                    successWRE(ieps,idt,ii) = norm((true_nz_weights~=0) - (XiWRMedian~=0))==0;
                    successWRE2(ieps,idt,ii) = norm((true_nz_weights~=0) - (XiWR~=0))==0;
                end
            end
            
            if runEL
                nWrongTermsDE(ieps,idt,ii) = sum(sum(abs((true_nz_weights~=0) - (XiD~=0))));
                modelErrorDE(ieps,idt,ii) = norm(XiD-true_nz_weights,"fro")/norm(true_nz_weights,"fro");
                successDE(ieps,idt,ii) = norm((true_nz_weights~=0) - (XiD~=0))==0;
                
                if runDoubleBag
                    nWrongTermsDDE(ieps,idt,ii) = sum(sum(abs((true_nz_weights~=0) - (XiDB~=0))));
                    nWrongTermsDDE2(ieps,idt,ii) = sum(sum(abs((true_nz_weights~=0) - (XiDBmed~=0))));
                    modelErrorDDE(ieps,idt,ii) = norm(XiDB-true_nz_weights,"fro")/norm(true_nz_weights,"fro");
                    modelErrorDDE2(ieps,idt,ii) = norm(XiDBmed-true_nz_weights,"fro")/norm(true_nz_weights,"fro");
                    successDDE(ieps,idt,ii) = norm((true_nz_weights~=0) - (XiDB~=0))==0;
                    successDDE2(ieps,idt,ii) = norm((true_nz_weights~=0) - (XiDBmed~=0))==0;
                end
                
                if runJK
                    nWrongTermsDEJK(ieps,idt,ii) = sum(sum(abs((true_nz_weights~=0) - (XiDJK~=0))));
                    modelErrorDEJK(ieps,idt,ii) = norm(XiDJK-true_nz_weights)/norm(true_nz_weights);
                    successDEJK(ieps,idt,ii) = norm((true_nz_weights~=0) - (XiDJK~=0))==0;
                end
            end
        end
        disp([num2str(idt) ' | ' num2str(ieps)]);
    end
    % if saveTrue
    %     save(sprintf('simOutE%.0f',ieps))
    % end
end

if saveTrue
    save('Rossler_simOutEFinal');
end

end


% % load data if not all cases are run for plot 
% if ~runS
%     load('results/simOutSFinalPaper');
% end
% warning('might load data ...')
% if ~runE
%     load('results/simOutEFinalPaper');
% end
% if ~runEL
%     load('results/simOutELFinalPaper');
% end

else
    
    load('results/simOutSFinalPaper');
    load('results/simOutEFinalPaper');
    load('results/simOutELFinalPaper');
    
end


%% plot results
% plotHeatMap
