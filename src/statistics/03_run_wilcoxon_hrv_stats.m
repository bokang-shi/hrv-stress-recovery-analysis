clear; clc;

% Paired Wilcoxon signed-rank tests for baseline vs stress HRV features.
%
% Expected input columns after MATLAB variable-name cleaning:
%   Subject, Base_RMSSD, Stress_RMSSD, Base_SDNN, Stress_SDNN, ...

inputFile = fullfile('outputs', 'wesad_hrv_sliding_median_ms2.csv');
outputDir = fullfile('outputs', 'statistics');
outputFile = fullfile(outputDir, 'wesad_hrv_wilcoxon_results.xlsx');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

L = readtable(inputFile);

metrics = {'RMSSD', 'SDNN', 'pNN50', 'LF_HF', 'LF_ms2', 'HF_ms2', 'LF_nu', 'HF_nu'};
nMetrics = numel(metrics);

raw_p  = nan(nMetrics, 1);
effect = nan(nMetrics, 1);
nPairs = zeros(nMetrics, 1);

for k = 1:nMetrics
    base_col   = ['Base_' metrics{k}];
    stress_col = ['Stress_' metrics{k}];

    if ~ismember(base_col, L.Properties.VariableNames) || ~ismember(stress_col, L.Properties.VariableNames)
        warning('Skipping %s because one or more columns were not found.', metrics{k});
        continue;
    end

    baseline = L.(base_col);
    stress   = L.(stress_col);

    valid = ~isnan(baseline) & ~isnan(stress);
    baseline = baseline(valid);
    stress   = stress(valid);
    nPairs(k) = numel(stress);

    if nPairs(k) == 0
        warning('Skipping %s because no paired observations were available.', metrics{k});
        continue;
    end

    delta = stress - baseline;
    effect(k) = median(delta);
    raw_p(k) = signrank(stress, baseline);
end

qvals = mafdr(raw_p, 'BHFDR', true);

Result_Wilcoxon = table( ...
    metrics', nPairs, effect, raw_p, qvals, qvals < 0.05, ...
    'VariableNames', {'Metric', 'N_Pairs', 'Median_Difference', 'Raw_p', 'BH_q', 'Significant'} ...
);

disp(Result_Wilcoxon);
writetable(Result_Wilcoxon, outputFile);
fprintf('\nSaved results to %s\n', outputFile);
