% Overlays network: Behrooz vs overlays benchmark. Compute theoretical efficiency.
% Please use Octave >= 3.8.1 for reasonable performances!

% Clear things up
clear all;
close all;

% Importing auxiliary functions
% source('gbnn_aux.m'); % does not work with MatLab, only Octave...
aux = gbnn_aux; % works with both MatLab and Octave

% Preparing stuff to automate the plots
% This will allow us to automatically select a different color and shape for each curve
colorvec = 'rgbmc';
markerstylevec = '+o*.xsd^v><ph';
linestylevec = {'-' ; ':' ; '--' ; '-.'};

% Vars config, tweak the stuff here
M = [0.01:0.05:0.09 0.1:0.1:0.4 0.5:0.25:4 5:1:6]; % this is a vector because we will try several values of m (number of messages, which influences the density)
%M = [0.005 5.1]; % to test both limits to check that the range is OK, the first point must be near 0 and the second point must be near 1, at least for one of the curves
Mcoeff = 1E3;
miterator = zeros(1,numel(M)); %M/2;
c = 8;
l = 16;
Chi = 16;
erasures = floor(c/2); %floor(c*0.25);
iterations = 4; % for convergence
tampered_messages_per_test = 30;
tests = 1;

enable_guiding = false;
gamma_memory = 1;
threshold = 0;
filtering_rule = 'GWsTA';
propagation_rule = 'sum';
tampering_type = 'erase';

residual_memory = 0;
filtering_rule_first_iteration = false;
filtering_rule_last_iteration = false;

% Overlays
enable_overlays = true;
overlays_max = [1 0];
%overlays_max = [1 5 20 100 1000 0];
overlays_interpolation = {'uniform'};

% Plot tweaking
statstries = 1; % retry n times with different networks to average (and thus smooth) the results
smooth_factor = 2; % interpolate more points to get smoother curves. Set to 1 to avoid smoothing (and thus plot only the point of the real samples).
smooth_method = 'cubic'; % use PCHIP or cubic to avoid interpolating into negative values as spline does
plot_curves_params = { 'markersize', 10, ...
                                            'linewidth', 1 ...
                                            };
plot_axis_params = { 'linewidth', 1, ...
                                      'tickdir', 'out', ...
                                      'ticklength', [0.01, 0.01] ...
                                      };
plot_text_params = { 'FontSize', 12, ... % in points
                                       'FontName', 'Helvetica' ...
                                       };

plot_theo = false; % plot theoretical error rates?
silent = false; % If you don't want to see the progress output
save_results = true; % save results to a file?

% == Launching the runs
D = zeros(numel(M), numel(overlays_max)*numel(overlays_interpolation));
E = zeros(numel(M), numel(overlays_max)*numel(overlays_interpolation));
EFF = zeros(numel(M), numel(overlays_max)*numel(overlays_interpolation));
TE = zeros(numel(M), numel(overlays_max)); % theoretical error rate depends on: Chi, l, c, erasures, enable_guiding and of course the density (theoretical or real) and thus on any parameter that changes the network (thus as the number of messages m to learn)

for t=1:statstries
    tperf = cputime(); % to show the total time elapsed later
    for m=1:numel(M) % and for each value of m, we will do a run

        counter = 1;
        for om=1:numel(overlays_max)
            cnetwork_stats = gbnn_theoretical_stats('Chi', Chi, 'c', c, 'l', l, 'M', M(m)*Mcoeff, 'erasures', erasures, 'overlays_max', overlays_max(om));

            % Store the results
            D(m,counter) = D(m,counter) + cnetwork_stats.theoretical_density;
            E(m,counter) = E(m,counter) + cnetwork_stats.theoretical_error_rate;
            EFF(m,counter) = EFF(m,counter) + cnetwork_stats.efficiency;
            TE(m, om) = cnetwork_stats.theoretical_error_rate;

            clear cnetwork_stats;

            counter = counter + 1;
        end
    end
    aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do all runs: %G seconds.\n'); aux.flushout(); % print total time elapsed
end
% Normalizing errors rates by calculating the mean error for all tries
D = D ./ statstries;
E = E ./ statstries;
EFF = EFF ./ statstries;
fprintf('END of all tests!\n'); aux.flushout();

% Print densities values and error rates
fprintf('Densities:\n'); disp(D);
fprintf('Error rates:\n'); disp(E);
fprintf('Efficiencies:\n'); disp(EFF);
fprintf('Theoretical error rates:\n'); disp(TE);
aux.flushout();

% == Plotting

% -- First interpolate data points to get smoother curves
% Note: if smooth_factor == 1 then these commands won't change the data points nor add more.
nsamples = numel(M);
M_interp = interp1(1:nsamples, M, linspace(1, nsamples, nsamples*smooth_factor), smooth_method);
D_interp = interp1(1:nsamples, D(:,1), linspace(1, nsamples, nsamples*smooth_factor), smooth_method);
E_interp = interp1(D(:,1), E, D_interp, smooth_method);
EFF_interp = interp1(D(:,1), EFF, D_interp, smooth_method);
TE_interp = interp1(D(:,1), TE, D_interp, smooth_method);

% -- Save results to a file
if save_results
    blacklist_vars = {'cnetwork', 's2', 'thriftymessages', 'currentpath', 'currentscriptname', 'outfile', 'blacklist_vars'}; % vars to NOT save because they are really to huge (several MB or even GB)

    % Prepare filepath, filename and mkdir
    [currentpath, currentscriptname] = fileparts(mfilename('fullpath'));
    outfile = sprintf('%s/results/%s.mat', currentpath, currentscriptname);
    fprintf('Saving results into results/%s\n', currentscriptname);
    if ~isequal(exist('results', 'dir'),7)
        mkdir('results');
    end

    % Write data to file in MATLAB format
    %save(outfile, 'results'); % save ALL the workspace into a file
    aux.savex(outfile, blacklist_vars{:}); % save ALL the workspace into a file except for a few variables which are just too big
end

% Plot error rate with respect to the density (or number of messages stored) and a few other parameters
figure; hold on;
xlabel(sprintf('(Bottom) Density  -- (Top) Number of stored messages (M) x%.1E', Mcoeff));
ylabel('Ratio efficiency / theoretical error rate');
counter = 1; % useful to keep track inside the matrix E. This is guaranteed to be OK since we use the same order of for loops (so be careful, if you move the forloops here in plotting you must also move them the same way in the tests above!)
for om=numel(overlays_max):-1:1
    colorcounter = om;
    if numel(overlays_interpolation) > 1; colorcounter = oi; end;
    coloridx = mod(colorcounter-1, numel(colorvec))+1; % change color if overlay or not

    % -- Set title
        plot_title = sprintf('%s', filtering_rule);
        if enable_guiding
            plot_title = strcat(plot_title, sprintf(' - Guided'));
        else
            plot_title = strcat(plot_title, sprintf(' - Blind'));
        end
        if overlays_max(om) == 1
            plot_title = strcat(plot_title, sprintf(' - One/No tags'));
        elseif overlays_max(om) == 0
            plot_title = strcat(plot_title, sprintf(' - M tags'));
        else
            plot_title = strcat(plot_title, sprintf(' - %i tags', overlays_max(om)));
        end

        % -- Efficiency 1
        lstyleidx = mod(counter-1, numel(linestylevec))+1; % change line style ...
        mstyleidx = mod(counter-1, numel(markerstylevec))+1; % and change marker style per plot

        lstyle = linestylevec(1, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
        ratioeff = zeros(size(E_interp(:,end+1-counter), 1), 1);
        E_plot = E_interp(:,end+1-counter);
        E_plot(E_plot == 0) = realmin();
        ratioeff = EFF_interp(:,end+1-counter) ./ E_plot;

        cur_plot = plot(D_interp, ratioeff, sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line
        set(cur_plot, plot_curves_params{:}); % additional plot style
        plot_title2 = strcat(plot_title, ' - ratio eff / err');

        set(cur_plot, 'DisplayName', plot_title2); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/

        % -- Efficiency 2
        lstyleidx = mod(counter-1, numel(linestylevec))+1; % change line style ...
        mstyleidx = mod(counter-1, numel(markerstylevec))+1; % and change marker style per plot

        lstyle = linestylevec(2, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
        ratioeff = zeros(size(E_interp(:,end+1-counter), 1), 1);
        ratioeff = (1 - E_interp(:,end+1-counter)) .* EFF_interp(:,end+1-counter);

        cur_plot = plot(D_interp, ratioeff, sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line
        set(cur_plot, plot_curves_params{:}); % additional plot style
        plot_title2 = strcat(plot_title, ' - ratio (1-eff) * err');

        set(cur_plot, 'DisplayName', plot_title2); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
        
        % -- Error rate
        lstyle = linestylevec(3, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
        cur_plot = plot(D_interp, E_interp(:,end+1-counter), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line
        set(cur_plot, plot_curves_params{:}); % additional plot style
        plot_title2 = strcat(plot_title, ' - theo error rate');

        set(cur_plot, 'DisplayName', plot_title2); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/


        counter = counter + 1;
end


% Plot theoretical error rates
if plot_theo
    %coloridx = mod(counter, numel(colorvec))+1;
    colornm = 'k';
    counter = 1;
    for om=numel(overlays_max):-1:1
        lstyleidx = mod(counter-1, numel(linestylevec))+1;
        mstyleidx = mod(counter-1, numel(markerstylevec))+1;

        lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
        cur_plot = plot(D_interp, TE_interp(:,om), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colornm)); % plot one line
        set(cur_plot, plot_curves_params{:}); % additional plot style

        plot_title = 'Theo. ';
        if enable_guiding
            plot_title = strcat(plot_title, sprintf(' - Guided'));
        else
            plot_title = strcat(plot_title, sprintf(' - Blind'));
        end
        if overlays_max(om) == 1
                plot_title = strcat(plot_title, sprintf(' - One/No tags'));
        elseif overlays_max(om) == 0
            plot_title = strcat(plot_title, sprintf(' - M tags'));
        else
            plot_title = strcat(plot_title, sprintf(' - %i tags', overlays_max(om)));
        end
        set(cur_plot, 'DisplayName', plot_title); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/

        counter = counter + 1;
    end
end

% Refresh plot with legends
legend(get(gca,'children'),get(get(gca,'children'),'DisplayName'), 'location', 'northwest'); % IMPORTANT: force refreshing to show the legend, else it won't show!
legend('boxoff');
% Add secondary axis on the top of the figure to show the number of messages
aux.add_2nd_xaxis(D(:,1), M, sprintf('x%.1E', Mcoeff), '%g', 0);
xlim([0 round(max(D(:,1)))]); % adjust x axis zoom
ylim([0 10]);
% Adjust axis drawing style
set( gca(), plot_axis_params{:} );
% Adjust text style
set([gca; findall(gca, 'Type','text')], plot_text_params{:});

% The end!
