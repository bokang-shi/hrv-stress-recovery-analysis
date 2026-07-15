clear; clc;

% Bootstrap confidence intervals for paired median HRV differences.
%
% Use a WESAD CSV for baseline vs stress only, or a primary experiment CSV
% containing baseline, stress, and recovery columns.

inputFile = fullfile('outputs', 'local_hrv_sliding_median_ms2.csv');
outputDir = fullfile('outputs', 'statistics');
sheetName = 1;
nBoot = 5000;
alpha = 0.05;

features = {'RMSSD', 'SDNN', 'LF/HF', 'pNN50', 'LF_ms2', 'HF_ms2'};

% {result title, later phase prefix, earlier phase prefix}
comparisons = {
    'Baseline vs Stress',  'Stress_',   'Base_';
    'Stress vs Recovery',  'Recovery_', 'Stress_'
};

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

[~,~,ext] = fileparts(inputFile);

switch lower(ext)
    case '.xlsx'
        data = readtable(inputFile, 'Sheet', sheetName, 'VariableNamingRule', 'preserve');
    case '.csv'
        data = readtable(inputFile, 'VariableNamingRule', 'preserve');
    otherwise
        error('Unsupported file type: %s', ext);
end

allRows = {};

for k = 1:size(comparisons, 1)
    resultName = comparisons{k, 1};
    prefix1 = comparisons{k, 2};
    prefix2 = comparisons{k, 3};

    resultsTable = analyseComparison(data, features, prefix1, prefix2, nBoot, alpha);

    fprintf('\n%s\n', resultName);
    disp(resultsTable);

    if ~isempty(resultsTable)
        resultsTable.Comparison = repmat(string(resultName), height(resultsTable), 1);
        allRows{end + 1} = resultsTable; %#ok<SAGROW>
    end
end

if ~isempty(allRows)
    combined = vertcat(allRows{:});
    combined = movevars(combined, 'Comparison', 'Before', 'Metric');
    outputFile = fullfile(outputDir, 'hrv_bootstrap_confidence_intervals.xlsx');
    writetable(combined, outputFile);
    fprintf('\nSaved results to %s\n', outputFile);
end

function resultsTable = analyseComparison(data, features, prefix1, prefix2, nBoot, alpha)

    Metric = {};
    N_Pairs = [];
    CI_Lower = [];
    CI_Upper = [];

    for i = 1:length(features)
        f = features{i};

        col1 = [prefix1 f];
        col2 = [prefix2 f];

        if ~ismember(col1, data.Properties.VariableNames)
            warning('Column "%s" not found. Skipping %s.', col1, f);
            continue;
        end
        if ~ismember(col2, data.Properties.VariableNames)
            warning('Column "%s" not found. Skipping %s.', col2, f);
            continue;
        end

        x1 = data.(col1);
        x2 = data.(col2);

        valid = ~isnan(x1) & ~isnan(x2);
        x1 = x1(valid);
        x2 = x2(valid);

        n = numel(x1);

        if n == 0
            warning('No valid paired data for %s (%s vs %s). Skipping.', f, prefix1, prefix2);
            continue;
        end

        diffVals = x1 - x2;

        bootStats = zeros(nBoot, 1);
        for b = 1:nBoot
            idx = randi(n, n, 1);
            bootStats(b) = median(diffVals(idx));
        end

        ci = prctile(bootStats, [100 * alpha / 2, 100 * (1 - alpha / 2)]);

        Metric{end + 1, 1} = f; %#ok<AGROW>
        N_Pairs(end + 1, 1) = n; %#ok<AGROW>
        CI_Lower(end + 1, 1) = ci(1); %#ok<AGROW>
        CI_Upper(end + 1, 1) = ci(2); %#ok<AGROW>
    end

    resultsTable = table(Metric, N_Pairs, CI_Lower, CI_Upper);
end
