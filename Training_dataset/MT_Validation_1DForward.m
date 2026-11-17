clc; clear; close all;

% 1. Load data
Forward = load('MT_Validation_1DForward.mat');
Fre = Forward.freq;

% 2. Define model indices and order for plotting
% Original order: 1=A, 2=Q, 3=K, 4=H
% We only want H-type (4) and K-type (3)
target_indices = [4, 3]; 
target_names   = {'H-type Model', 'K-type Model'};
% Corresponding resistivity parameters extracted by index
all_rho_vals   = { [10, 100, 1000], [1000, 100, 10], [10, 1000, 100], [100, 10, 1000] };
h_info = 'h_1 = 2 km, h_2 = 5 km';

% 3. Create figure (width reduced from 1200 to 900 because only two columns)
figure('Color', 'w', 'Units', 'pixels', 'Position', [100, 100, 900, 700]);

% Loop over the two models
for k = 1:length(target_indices)
    
    % Get the actual index in the original data (4 or 3)
    real_idx = target_indices(k);
    
    % ==================== Plot apparent resistivity (first row) ====================
    subplot(2, 2, k); 
    
    % 1D analytical solution - black solid line + circles + black fill
    h1 = loglog(Fre, Forward.rho_ana(:, real_idx), 'k-o', ...
        'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', 'k'); hold on;
    
    % 2D FDM TE mode - blue triangles + blue fill
    h2 = loglog(Fre, Forward.rho_te(:, real_idx), 'b^', ...
        'MarkerSize', 5, 'MarkerFaceColor', 'b');
    
    % 2D FDM TM mode - red squares + red fill
    h3 = loglog(Fre, Forward.rho_tm(:, real_idx), 'rs', ...
        'MarkerSize', 5, 'MarkerFaceColor', 'r');
    
    % --- Axis beautification ---
    set(gca, 'XDir', 'reverse', 'FontSize', 12, 'FontName', 'Times New Roman');
    grid on; grid minor;
    
    % Add title (to distinguish H and K)
    current_rhos = all_rho_vals{real_idx};
    
    % Show y-label and legend only for the first column
    if k == 1
        ylabel('Apparent Resistivity (\Omega\cdotm)', 'FontSize', 12, 'FontWeight', 'bold');
        % Legend placed at lower left or best location
        legend([h1, h2, h3], {'1D Analytical solution', '2D FDM TE solution', '2D FDM TM solution'}, ...
               'Location', 'SouthWest', 'FontSize', 10); 
    end
    xlabel('Frequency (Hz)', 'FontSize', 12);
    
    % ==================== Plot phase (second row) ====================
    subplot(2, 2, k + 2);
    
    % 1D analytical solution - black solid line + circles + black fill
    semilogx(Fre, Forward.phs_ana(:, real_idx), 'k-o', ...
        'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', 'k'); hold on;
    
    % 2D FDM TE mode - blue triangles + blue fill
    semilogx(Fre, Forward.phs_te(:, real_idx), 'b^', ...
        'MarkerSize', 5, 'MarkerFaceColor', 'b');
    
    % 2D FDM TM mode - red squares + red fill
    semilogx(Fre, Forward.phs_tm(:, real_idx), 'rs', ...
        'MarkerSize', 5, 'MarkerFaceColor', 'r');
    
    set(gca, 'XDir', 'reverse', 'FontSize', 12, 'FontName', 'Times New Roman');
    grid on; grid minor;
    ylim([0, 90]); 
    yticks(0:15:90);
    
    if k == 1
        ylabel('Phase (degree)', 'FontSize', 12, 'FontWeight', 'bold');
    end
    xlabel('Frequency (Hz)', 'FontSize', 12);
end
