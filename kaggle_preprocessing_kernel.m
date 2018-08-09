% The dataset of our work where the application of portable Raman
% spectroscopy coupled with several supervised machine-learning techniques,
% is used to discern between diabetic patients (DM2) and healthy controls
% (Ctrl), with a high degree of accuracy.
% This script pre-processes the spectra to reproduce Fig. 1 of our paper:
% 
% Use of Raman Spectroscopy to Screen Diabetes Mellitus with Machine
% Learning Tools 
% Edgar Guevara, Juan Carlos Torres-Galván, Miguel G. Ramírez-Elías,
% Claudia Luevano-Contreras and Francisco Javier González
% Biomedical Optics Express (2018)
% _______________________________________________________________________________
% Copyright (C) 2018 Edgar Guevara, PhD
% CONACYT-Universidad Autónoma de San Luis Potosí
% Coordinación para la Innovación y Aplicación de la Ciencia y la Tecnología
% _______________________________________________________________________________
%% Read data from csv files
clear; close all; clc
innerArm_dataOut = import_csv_file('innerArm.csv');
innerArmData = innerArm_dataOut{2:end, 3:end};
ear_dataOut = import_csv_file('earLobe.csv');
earData = ear_dataOut{2:end, 3:end};
nail_dataOut = import_csv_file('thumbNail.csv');
nailData = nail_dataOut{2:end, 3:end};
vein_dataOut = import_csv_file('vein.csv');
veinData = vein_dataOut{2:end, 3:end};
AllWaveNumber = innerArm_dataOut{1,3:end};
has_DM2 = innerArm_dataOut.has_DM2(2:end);
has_DM2 = has_DM2 == categorical(1);
labelsText = innerArm_dataOut.patientID(2:end);

%% Crop Data to 800-1800 cm^-1 (max. wavenumber 3159cm^-1)
waveNumbers2Crop = [800 1800];
idxCrop = (find(AllWaveNumber==waveNumbers2Crop(1)) : find(AllWaveNumber==waveNumbers2Crop(2)))';
waveNumber      = AllWaveNumber(idxCrop);
innerArmData    = innerArmData(:,idxCrop);
earData         = earData(:,idxCrop);
nailData        = nailData(:, idxCrop);
veinData        = veinData(:, idxCrop);

%% Apply vancouver fluorescence removal and normalization
% Download code to remove fluorescence from
% https://github.com/guevaracodina/vancouver
polyOrder = 5;
errThreshold = 0.05;
nPoints = 5;
nIter = 100;
for iSamples = 1:size(labelsText,1)
    % ----------------------- innerArm -------------------------------------
    raman = innerArmData(iSamples, :);    % Get sample
    raman = removeNaN(raman);           % Remove NaN's
    % Remove fluorescence
    [~, raman, waveNumber] = vancouver(waveNumber, raman,...
                          polyOrder, errThreshold, nPoints, nIter);
    raman = raman';                  
	raman = removeNaN(raman);           % Remove NaN's
	% Normalize data (AUC=1)
    raman = norm_auc( waveNumber, raman);
    % Normalize min-max to 0 & 1 (for plotting purposes only)
    raman = raman - min(raman);
    raman = raman ./ max(raman);
	innerArmData(iSamples, :) = raman;    % Replace sample
    % ------------------------- Ear --------------------------------------
    raman = earData(iSamples, :);    % Get sample
    raman = removeNaN(raman);           % Remove NaN's
    % Remove fluorescence
    [~, raman, waveNumber] = vancouver(waveNumber, raman,...
                          polyOrder, errThreshold, nPoints, nIter);
	raman = raman';
	raman = removeNaN(raman);           % Remove NaN's
	% Normalize data (AUC=1)
    raman = norm_auc( waveNumber, raman);
    % Normalize min-max to 0 & 1 (for plotting purposes only)
    raman = raman - min(raman);
    raman = raman ./ max(raman);
	earData(iSamples, :) = raman;    % Replace sample
    % ------------------------- Nail -------------------------------------
    raman = nailData(iSamples, :);    % Get sample
    raman = removeNaN(raman);           % Remove NaN's
    % Remove fluorescence
    [~, raman, waveNumber] = vancouver(waveNumber, raman,...
                          polyOrder, errThreshold, nPoints, nIter);
	raman = raman';
	raman = removeNaN(raman);           % Remove NaN's
	% Normalize data (AUC=1)
    raman = norm_auc( waveNumber, raman);
    % Normalize min-max to 0 & 1 (for plotting purposes only)
    raman = raman - min(raman);
    raman = raman ./ max(raman);
	nailData(iSamples, :) = raman;    % Replace sample
    % ------------------------- Vein -------------------------------------
    raman = veinData(iSamples, :);    % Get sample
    raman = removeNaN(raman);           % Remove NaN's
    % Remove fluorescence
    [~, raman, waveNumber] = vancouver(waveNumber, raman,...
                          polyOrder, errThreshold, nPoints, nIter);
	raman = raman';
    raman = removeNaN(raman);           % Remove NaN's
	% Normalize data (AUC=1)
    raman = norm_auc( waveNumber, raman);
    % Normalize min-max to 0 & 1 (for plotting purposes only)
    raman = raman - min(raman);
    raman = raman ./ max(raman);
	veinData(iSamples, :) = raman;    % Replace sample
end


%% Center data
NormFlag = false; % zscore normalization
if NormFlag
    innerArmData = zscore(innerArmData);
    earData     = zscore(earData);
    nailData    = zscore(nailData);
    veinData    = zscore(veinData);
end

%% Offset spectra
% Shift data (inner arm)
DM2tmp = innerArmData(has_DM2,:);
DM2tmp = DM2tmp(:);
ctrltmp = innerArmData(~has_DM2,:);
ctrltmp = ctrltmp(:);
innerArmData(~has_DM2,:) = innerArmData(~has_DM2,:) + max(DM2tmp) + std(DM2tmp) + std(ctrltmp);
% Shift data (earlobe)
DM2tmp = earData(has_DM2,:);
DM2tmp = DM2tmp(:);
ctrltmp = earData(~has_DM2,:);
ctrltmp = ctrltmp(:);
earData(~has_DM2,:) = earData(~has_DM2,:) + max(DM2tmp) + std(DM2tmp) + std(ctrltmp);
% Shift data (thumbnail)
DM2tmp = nailData(has_DM2,:);
DM2tmp = DM2tmp(:);
ctrltmp = nailData(~has_DM2,:);
ctrltmp = ctrltmp(:);
nailData(~has_DM2,:) = nailData(~has_DM2,:) + max(DM2tmp) + std(DM2tmp) + std(ctrltmp);
% Shift data (vein)
DM2tmp = veinData(has_DM2,:);
DM2tmp = DM2tmp(:);
ctrltmp = veinData(~has_DM2,:);
ctrltmp = ctrltmp(:);
veinData(~has_DM2,:) = veinData(~has_DM2,:) + max(DM2tmp) + std(DM2tmp) + std(ctrltmp);

%% Plot data
% Download code to plot shaded error bars from 
% https://la.mathworks.com/matlabcentral/fileexchange/26311-raacampbell-shadederrorbar
close all
figure; set(gcf, 'color', 'w'); hold on
yLimits = [-0.05 2.5];
myLineWidth = 1.5;

% ----------------------- innerArm -------------------------------------
subplot(221)
myColor = [1 0 0];
H(1) = shadedErrorBar(waveNumber', innerArmData(has_DM2,:), {@mean, @std},...
    {'Color', myColor, 'LineWidth', myLineWidth}, 1);
myColor = [0 0 1];
hold on
H(2) = shadedErrorBar(waveNumber', innerArmData(~has_DM2,:), {@mean, @std},...
    {'Color', myColor, 'LineWidth', myLineWidth}, 1);
% axis off
axis tight
ylim(yLimits);
title('Inner Arm', 'FontSize', 14);
ylabel('Raman Intensity (a.u.)', 'FontSize', 14);
xlabel('Raman Shift (cm^{-1})', 'FontSize', 14);
legend([H(1).mainLine, H(1).patch(1), H(2).mainLine, H(2).patch(1)], ...
    {'DM2' 'DM2 \sigma' 'Ctrl' 'Ctrl \sigma'}, ...
    'Location', 'NorthEast', 'FontSize', 14);
set(gca, 'FontSize', 14)

% ----------------------- Ear -------------------------------------
subplot(222)
myColor = [1 0 0];
H(1) = shadedErrorBar(waveNumber', earData(has_DM2,:), {@mean, @std},...
    {'Color', myColor, 'LineWidth', myLineWidth}, 1);
myColor = [0 0 1];
hold on
H(2) = shadedErrorBar(waveNumber', earData(~has_DM2,:), {@mean, @std},...
    {'Color', myColor, 'LineWidth', myLineWidth}, 1);
% axis off
axis tight
ylim(yLimits);
title('Ear Lobe', 'FontSize', 14);
ylabel('Raman Intensity (a.u.)', 'FontSize', 14);
xlabel('Raman Shift (cm^{-1})', 'FontSize', 14);
legend([H(1).mainLine, H(1).patch(1), H(2).mainLine, H(2).patch(1)], ...
    {'DM2' 'DM2 \sigma' 'Ctrl' 'Ctrl \sigma'}, ...
    'Location', 'NorthEast', 'FontSize', 14);
set(gca, 'FontSize', 14)

% ----------------------- Nail -------------------------------------
subplot(223)
myColor = [1 0 0];
H(1) = shadedErrorBar(waveNumber', nailData(has_DM2,:), {@mean, @std},...
    {'Color', myColor, 'LineWidth', myLineWidth}, 1);
myColor = [0 0 1];
hold on
H(2) = shadedErrorBar(waveNumber', nailData(~has_DM2,:), {@mean, @std},...
    {'Color', myColor, 'LineWidth', myLineWidth}, 1);
% axis off
axis tight
ylim(yLimits);
title('Thumb Nail', 'FontSize', 14);
ylabel('Raman Intensity (a.u.)', 'FontSize', 14);
xlabel('Raman Shift (cm^{-1})', 'FontSize', 14);
legend([H(1).mainLine, H(1).patch(1), H(2).mainLine, H(2).patch(1)], ...
    {'DM2' 'DM2 \sigma' 'Ctrl' 'Ctrl \sigma'}, ...
    'Location', 'NorthEast', 'FontSize', 14);
set(gca, 'FontSize', 14)

% ----------------------- Vein -------------------------------------
subplot(224)
myColor = [1 0 0];
H(1) = shadedErrorBar(waveNumber', veinData(has_DM2,:), {@mean, @std},...
    {'Color', myColor, 'LineWidth', myLineWidth}, 1);
myColor = [0 0 1];
hold on
H(2) = shadedErrorBar(waveNumber', veinData(~has_DM2,:), {@mean, @std},...
    {'Color', myColor, 'LineWidth', myLineWidth}, 1);
% axis off
axis tight
ylim(yLimits);
title('Cubital Vein', 'FontSize', 14);
ylabel('Raman Intensity (a.u.)', 'FontSize', 14);
xlabel('Raman Shift (cm^{-1})', 'FontSize', 14);
legend([H(1).mainLine, H(1).patch(1), H(2).mainLine, H(2).patch(1)], ...
    {'DM2' 'DM2 \sigma' 'Ctrl' 'Ctrl \sigma'}, ...
    'Location', 'NorthEast', 'FontSize', 14);
set(gca, 'FontSize', 14)

%% Change figure aspect for printing
% 10 in wide x 5 in tall @ 1200 dpi
% Change figure and paper size
set(gcf, 'units','inches', 'position',[0.1 0.1 11 4.5], 'PaperPosition', [0.1 0.1 11 4.5])

% Print figure
figFolder = 'C:\Edgar\Dropbox\CIACYT\Students\Juan Carlos\Figures';
if (false)
    % Save as PNG at the user-defined resolution
    print(gcf, '-dpng', ...
        fullfile(figFolder, 'raman_spectra_ctrl_mean.png'),...
        sprintf('-r%d',1200));
end

% EOF
