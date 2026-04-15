clear; clc;

filename = 'wesad_hrv_sliding_median_ms2.csv';   % e.g. 'experiment data.xlsx' or 'wesad_hrv_sliding_median_ms2.csv'
sheetName = 1;               
nBoot = 5000;
alpha = 0.05;

features = {'RMSSD','SDNN','LF/HF','pNN50','LF_ms2','HF_ms2'};

% Define comparisons: {result title, prefix1, prefix2}
comparisons = {'Baseline vs Stress',  'Stress_', 'Base_';'Stress vs Recovery', 'Recovery_', 'Stress_'};

[~,~,ext] = fileparts(filename);

switch lower(ext)
    case '.xlsx'
        data = readtable(filename, 'Sheet', sheetName, 'VariableNamingRule', 'preserve');
    case '.csv'
        data = readtable(filename, 'VariableNamingRule', 'preserve');
    otherwise
        error('Unsupported file type: %s', ext);
end

allResults = struct();

for k = 1:size(comparisons, 1)
    resultName = comparisons{k,1};
    prefix1 = comparisons{k,2};
    prefix2 = comparisons{k,3};

    resultsTable = analyseComparison(data, features, prefix1, prefix2, nBoot, alpha);

    fieldName = matlab.lang.makeValidName(resultName);
    allResults.(fieldName) = resultsTable;

    fprintf('\n %s \n', resultName);
    disp(resultsTable);
end

function resultsTable = analyseComparison(data, features, prefix1, prefix2, nBoot, alpha)

    Metric = {};
    CI_Lower = [];
    CI_Upper = [];

    for i = 1:length(features)
        f = features{i};

        col1 = [prefix1 f];
        col2 = [prefix2 f];

        % Check columns exist
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

        % Keep paired non-NaN observations
        valid = ~isnan(x1) & ~isnan(x2);
        x1 = x1(valid);
        x2 = x2(valid);

        n = numel(x1);

        if n == 0
            warning('No valid paired data for %s (%s vs %s). Skipping.', f, prefix1, prefix2);
            continue;
        end

        diffVals = x1 - x2;

        % Bootstrap CI for median difference
        bootStats = zeros(nBoot,1);
        for b = 1:nBoot
            idx = randi(n, n, 1);
            bootStats(b) = median(diffVals(idx));
        end

        ci = prctile(bootStats, [100*alpha/2, 100*(1-alpha/2)]);

        Metric{end+1,1} = f;
        CI_Lower(end+1,1) = ci(1);
        CI_Upper(end+1,1) = ci(2);
    end
    resultsTable = table(Metric, CI_Lower, CI_Upper);
end