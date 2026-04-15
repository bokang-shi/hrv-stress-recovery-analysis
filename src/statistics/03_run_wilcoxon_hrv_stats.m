clear; clc;

L = readtable('data.csv');

metrics = {'RMSSD', 'SDNN', 'pNN50', 'LF_HF','LF_ms2', 'HF_ms2', 'LF_nu', 'HF_nu'};

nMetrics = numel(metrics);

raw_p  = zeros(nMetrics,1);
effect = zeros(nMetrics,1);
%% 2. Wilcoxon signed-rank test (paired)
for k = 1:nMetrics

    base_col   = ['Base_'   metrics{k}];
    stress_col = ['Stress_' metrics{k}];

    baseline = L.(base_col);
    stress   = L.(stress_col);

    % Remove NaNs
    valid = ~isnan(baseline) & ~isnan(stress);
    baseline = baseline(valid);
    stress   = stress(valid);

    % Paired difference
    delta = stress - baseline;

    % Effect size (median difference, recommended for Wilcoxon)
    effect(k) = median(delta);

    % Wilcoxon signed-rank test
    raw_p(k) = signrank(stress, baseline);  
end
%% 3. BH (FDR) correction
qvals = mafdr(raw_p, 'BHFDR', true);

Result_Wilcoxon = table( metrics', effect, raw_p, qvals, qvals < 0.05, 'VariableNames', {'Metric', 'Median_Difference','Raw_p', 'BH_q', 'Significant'} );

disp(Result_Wilcoxon)

writetable(Result_Wilcoxon, 'wesad_hrv_wilcoxon_results.xlsx');
