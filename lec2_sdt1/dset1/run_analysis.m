% Run the published OSF analysis headlessly and export all generated figures.

project_root = fileparts(mfilename('fullpath'));
source_dir = fullfile(project_root, 'osfstorage', 'Code', 'Source Data');
code_dir = fullfile(project_root, 'osfstorage', 'Code');

assert(isfolder(source_dir), 'Source-data directory not found: %s', source_dir);
assert(license('test', 'Optimization_Toolbox') == 1, ...
    'MATLAB Optimization Toolbox is unavailable.');

cd(source_dir);
set(groot, 'DefaultFigureVisible', 'off');
addpath(code_dir);
plotres;

figures = findall(groot, 'Type', 'figure');
assert(numel(figures) == 6, 'Expected six figures, generated %d.', numel(figures));

project_root = fileparts(mfilename('fullpath'));
results_dir = fullfile(project_root, 'results');
if ~isfolder(results_dir)
    mkdir(results_dir);
end

exportgraphics(fig2a, fullfile(results_dir, 'Figure_2a.png'), 'Resolution', 150);
exportgraphics(fig2b, fullfile(results_dir, 'Figure_2b.png'), 'Resolution', 150);
exportgraphics(fig3, fullfile(results_dir, 'Figure_3.png'), 'Resolution', 150);
exportgraphics(figS1a, fullfile(results_dir, 'Figure_S1a.png'), 'Resolution', 150);
exportgraphics(figS1b, fullfile(results_dir, 'Figure_S1b.png'), 'Resolution', 150);
exportgraphics(figS2, fullfile(results_dir, 'Figure_S2.png'), 'Resolution', 150);

fprintf('OSF analysis completed; generated and exported 6 figures to %s.\n', ...
    results_dir);

