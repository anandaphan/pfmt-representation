function export_csvs(arg1, arg2)
    path = arg1;
    load(path);
    baseDir = arg2;  % Root folder for all exports
    mkdir(baseDir);

    numCases = size(MTOData.Results, 1);
    numAlgs  = size(MTOData.Results, 2);
    numRuns  = size(MTOData.Results, 3);

    for i = 1:numCases
        for j = 1:numAlgs
            % Create folder for each Case and Algorithm
            caseFolder = fullfile(baseDir, sprintf('Case_%02d', i));
            algFolder  = fullfile(caseFolder, MTOData.Algorithms(j).Name);
            if ~exist(algFolder, 'dir')
                mkdir(algFolder);
            end

            for k = 1:numRuns
                % Extract and transpose the data
                data = MTOData.Results(i,j,k).Obj';

                % Generate filename
                filename = fullfile(algFolder, sprintf('Run_%02d.csv', k));

                % Write to CSV
                writematrix(data, filename);
            end
        end
    end

    disp('All CSV files exported successfully!');
end