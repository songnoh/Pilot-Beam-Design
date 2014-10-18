% Source Code (in part) for pilot beam pattern design algorithms
% May 14, 2014
clc; clear all;
reset(RandStream.getGlobalStream,sum(100*clock));
jj = sqrt(-1);


Nt=2^8;  Nr=1;                      % number of antennas
M = 10;                             % block transmission with lenght M=Mp+Md; pilot/data transmission
indPilot = [1,2];                   % pilot symbol time indices
indData  = setdiff(1:M,indPilot);   % data symbol time indices
Mp = length(indPilot);              % length of training period
Md = length(indData);               % length of data transmission period

numSymbol=floor(Nt/length(indPilot)+2)*M; % symbol time duration for simulation
numIter = 100;                              % number of Monte Carlo iteration
transmitPwr = 15;                         % transmit power (dB)
noisePwr = 1/(10^(transmitPwr/10));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%* Generate channel spatial correlation (one-ring) *%
h = 60;         % an elevation of the center of the cell (meter)
radius = 30;    % the radius of scattering rings model (meter)
distance = 100; % distance from the base station (meter)
D = 1/2;        % half-wavelength antenna spacing 

pl_exp = 3.8;   % path loss exponent
pl_ref = 30;    % path loss reference distance (m)
pl = 1/(1+(distance/pl_ref)^pl_exp); % path loss

% Nt_V x Nt_H rectangular antenna at base station
Nt_H = 2^4; Nt_V = 2^4;
if all(Nt - Nt_H*Nt_V), error('The number of rectangular arrays does not match.');end

fileName = sprintf('oneringchannel_Nt%d(%dx%d)_h%d_r%d_d%d',Nt,Nt_H,Nt_V,h,radius,distance);
try
    load(fileName);
catch exception
    if ~exist(fileName,'file')
        fprintf('One-ring Channel Based Spatial Covariance (Nt%d: %dx%d): \n',Nt,Nt_H,Nt_V);
        syms alpha;

        % Horizontal channel spatial covariance
        delta_H = atan(radius/distance);
        theta_H = pi/6; % [-pi/3, pi/3]
        col_H = zeros(Nt_H,1);
        for n=1:Nt_H
            m=1;
            f = 1/(2*delta_H) * exp(-jj*2*pi*D*(n-m)*sin(alpha));
            F = int(f,alpha,theta_H-delta_H,theta_H+delta_H);
            col_H(n) = double(F);
        end

        % Vertical channel spatial covariance
        delta_V = 1/2 * (atan((distance+radius)/h) - atan((distance-radius)/h));
        theta_V = 1/2 * (atan((distance+radius)/h) + atan((distance-radius)/h));
        col_V = zeros(Nt_V,1);
        for n=1:Nt_V
            m=1;
            f = 1/(2*delta_V) * exp(-jj*2*pi*D*(n-m)*sin(alpha));
            F = int(f,alpha,theta_V-delta_V,theta_V+delta_V);
            col_V(n) = double(F);
        end

        R_H = toeplitz(col_H,conj(col_H));
        R_V = toeplitz(col_V,conj(col_V));

        % Check positive semidefiniteness of covariance matrices
        posR_H = all(eig(R_H) >= 0);
        posR_V = all(eig(R_V) >= 0);
        if ~posR_H,    R_H = R_H + abs(min(eig(R_H)))*eye(Nt_H);    end
        if ~posR_V,    R_V = R_V + abs(min(eig(R_V)))*eye(Nt_V);    end

        Rh = pl*kron(R_H,R_V);
        save(sprintf(fileName),'Rh');
    end
end
[uRh,sRh,vRh] = svd(Rh);
sh=diag(sRh);



%* State-space system parameter *%
% - At carrier frequency 2.5GHz, e.g.,
a = 0.999995233766144; % temporal correlation
A = a*eye(Nt);
B = sqrt(1-a^2)*uRh*sqrt(sRh)*vRh';
W = (1-a^2)*Rh;
V = eye(Nr)*noisePwr;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%* Pilot beam pattern configuration *%

Methods = {'Orthogonal','Random','M_p eigenvectors','Algorithm 1','Perfect CSIT'};
% 1) Orthogonal pilot beams, 
% 2) Randomly generated pilot beams, 
% 3) Mp eigenvectors of spatial covariance matrix, 
% 4) Proposed:Sequential, 
% 5) Perfect CSIT
numKind = length(Methods);
C = cell(numKind,numKind);

Orth_pilot = eye(Nt); % hadamard(Nt)/sqrt(Nt);
for i=1:Nt
    C{1,i} = Orth_pilot(:,i);
    C{2,i} = randn(Nt,1); C{2,i}=C{2,i}/norm(C{2,i},'fro'); % power normalization
    C{3,i} = uRh(:,i);
    C{4,i} = uRh(:,i);
    C{5,i} = zeros(Nt,1);
end




%% MAIN CODE
traceP = zeros(numKind,numSymbol);
nmseh  = zeros(numKind,numSymbol);
recSNR = zeros(numKind,numSymbol);
y      = cell(numKind,1);
K      = cell(numKind,1);
h_m    = cell(numKind,1);
h_k    = cell(numKind,1);
P_m    = cell(numKind,1);
P_k    = cell(numKind,1);

fprintf('Run....\n');
for indIter=1:numIter
    tic;
    
    % Initialization
    h = uRh*sqrt(sRh)*(randn(Nt,1)+sqrt(-1)*randn(Nt,1))/sqrt(2);     
    for indK=1:numKind
        h_m{indK} = zeros(Nt,1);   % h_k|k-1
        P_m{indK} = Rh;            % P_k|k-1
    end

    s = sh;
    IND_OrthRandom = 1;
    IND_MpEigVec = 1;
    
    for indSymbol=1:numSymbol
        h = a*h + B*(randn(Nt,1)+sqrt(-1)*randn(Nt,1))/sqrt(2);     % Gauss-Markov Process
        noise = sqrt(noisePwr)*(randn(Nr,1)+sqrt(-1)*randn(Nr,1))/sqrt(2);

        if ismember(mod(indSymbol-1,M)+1,indPilot)
            
            % PILOT PERIOD %
            for indK=1:numKind
                switch indK
                    case {1,2} % Orthogonal & Random
                        IND = floor(mod(IND_OrthRandom-1,Nt))+1;
                        IND_OrthRandom = IND_OrthRandom + 0.5;
                    case 3     % Mp eigenvectors
                        IND = mod(IND_MpEigVec-1,Mp)+1;
                        IND_MpEigVec = IND_MpEigVec+1;
                    case 4     % Proposed (sequential)
                        [mV,mI] = max(s);
                        s(mI)=(s(mI)*noisePwr)/(s(mI)+noisePwr);
                        s = a^2*s+(1-a^2)*sh;
                        IND = mI;
                    case 5     % Perfect CSIT
                    otherwise, error('Not defined method.');
                end
                
                y{indK}   = C{indK,IND}'*h + noise; % received signal
                
                % Kalman filtering: Measurement update
                K{indK}   = P_m{indK}*C{indK,IND}/(C{indK,IND}'*P_m{indK}*C{indK,IND}+V);
                h_k{indK} = h_m{indK} +K{indK}*(y{indK}-C{indK,IND}'*h_m{indK});   % h_k|k
                P_k{indK} = (eye(Nt) - K{indK}*C{indK,IND}')*P_m{indK};            % P_k|k

                h_k{numKind}=h; P_k{numKind}=zeros(Nt); % Perfect CSIT case
                
                % Kalman filtering: Time update
                P_m{indK} = a*P_k{indK}*a' + W;    % P_k+1|k
                h_m{indK} = a*h_k{indK};           % h_k+1|k

                traceP(indK,indSymbol) = traceP(indK,indSymbol) + real(trace(P_k{indK}))/numIter;
                nmseh(indK,indSymbol)  = nmseh(indK,indSymbol)  + (norm(h_k{indK}-h)/norm(h))^2/numIter;
                recSNR(indK,indSymbol) = recSNR(indK,indSymbol) + NaN;
            end
        else
            % DATA PERIOD %
            s = a^2*s+(1-a^2)*sh;
            
            for indK=1:numKind
                
                traceP(indK,indSymbol) = traceP(indK,indSymbol) + real(trace(P_m{indK}))/numIter;
                nmseh(indK,indSymbol)  = nmseh(indK,indSymbol)  + (norm(h_m{indK}-h)/norm(h))^2/numIter;
                
                w = h_m{indK}/norm(h_m{indK});     % unit norm beamforming vector
                recSNR(indK,indSymbol) = recSNR(indK,indSymbol) + ...
                                        real( norm(h_m{indK})^2 / (w'*P_m{indK}*w+noisePwr) )/numIter;

                % Kalman filtering: Time update
                P_m{indK} = a*P_m{indK}*a' + W;    % P_k+1|k
                h_m{indK} = a*h_m{indK};           % h_k+1|k
            end
        end
    end
    
    fprintf('Iter=%2.0f: ', indIter);
    toc;
end



%% DRAW FIGURE
mSize=6;    lWidth=1;   fSize=15;
mark = {'+','x','s',' ','.'};
line = {' ',' ',' ','-',' '};
color = {'k','k','k','r','m'};
Xaxis = 1:100; % Specify a subrange of X-axis for illustration

figure(1);
% FIGURE: NMSE (empirical)
subplot(3,1,1);
for i=1:length(Methods)
    plot(Xaxis,nmseh(i,Xaxis),strcat(mark{i},line{i},color{i}),'LineWidth',lWidth,'MarkerSize',mSize); 
    if i==1, hold on; end
end
hold off;
legend(Methods,'Location','East');
xlabel('Time (k)','FontSize',fSize);
ylabel('NMSE (empirical)','FontSize',fSize);
set(gca,'FontSize',fSize);
axis([Xaxis(1) Xaxis(end) 0 1]);


% FIGURE: NMSE using Kalman matrix
subplot(3,1,2);
for i=1:length(Methods)
    plot(Xaxis,traceP(i,Xaxis)/sum(sh),strcat(mark{i},line{i},color{i}),'LineWidth',lWidth,'MarkerSize',mSize); 
    if i==1, hold on; end
end
hold off;
xlabel('Time (k)','FontSize',fSize);
ylabel('NMSE: tr(P_{k|k})/tr(R_h)','FontSize',fSize)
set(gca,'FontSize',fSize);
axis([Xaxis(1) Xaxis(end) 0 1]);


% FIGURE: Received SNR
subplot(3,1,3);
for i=1:length(Methods)
    plot(Xaxis,10*log10(recSNR(i,Xaxis)),strcat(mark{i},line{i},color{i}),'LineWidth',lWidth,'MarkerSize',mSize); 
    if i==1, hold on; end
end
hold off;
xlabel('Time (k)','FontSize',fSize);
ylabel('Received SNR','FontSize',fSize)
set(gca,'FontSize',fSize);
yAxisLim = [min(min(10*log10(recSNR))), max(max(10*log10(recSNR)))];
axis([Xaxis(1) Xaxis(end) yAxisLim]);
