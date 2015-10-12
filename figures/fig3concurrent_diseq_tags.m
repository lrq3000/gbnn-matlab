% This example main file shows how to reproduce the figure 3 of the 2014 article but with concurrent messages. This is mainly used to analyze and compare the performances of the different filtering_rules when concurrency is enabled.

% Clear things up
clear all;
close all;

% Addpath of the whole library (this allows for modularization: we can place the core library into a separate folder)
if ~exist('gbnn_aux.m','file')
    %restoredefaultpath;
    addpath(genpath(strcat(cd(fileparts(mfilename('fullpath'))),'/../gbnn-core/')));
end

% Importing auxiliary functions
% source('gbnn_aux.m'); % does not work with MatLab, only Octave...
aux = gbnn_aux; % works with both MatLab and Octave

% Fix issues when trying to print into an eps.
% set (0, 'defaultaxesfontname', 'Helvetica'); % fix GhostScript error about handling unknown fonts
% graphics_toolkit('gnuplot'); % fix truncated curves in output file

% Preparing stuff to automate the plots
% This will allow us to automatically select a different color and shape for each curve
colorvec = 'krbgmc';
markerstylevec = '+o*.xsd^v><ph';
linestylevec = {'-' ; '--' ; ':' ; '-.'};

% Vars config, tweak the stuff here
%M = [0.1:0.2:1.5 1.8 2:1:11 15 40]; % this is a vector because we will try several values of m (number of messages, which influences the density)
M = [0.1:0.2:1.5 1.8 2:1:11];
Mcoeff = 1E3;
miterator = zeros(1,numel(M)); %M/2;
c = 8;
l = 16;
Chi = 32;
erasures = 2;
iterations = 4; % must be > 1 for disequilibrium to take effect. Note: useless for filtering rule ML, therefore it will be automatically set to 1 iiteration only for ML, the other filtering_rules will use the number of iterations you specify here.
tampered_messages_per_test = 30; % aka sampling rate, the size of the sample to compute the retrieval error rate (higher = more precise, eg if 30 it means that percentage will be in increment of 3.333...%, there cannot be 2.5% for example)
tests = 1; % number of tests for the retrieval error rate (this will redo only the testing phase, not the learning phase. In the end, it's just a multiplier to the sampling rate, aka tampered_messages_per_test.

enable_guiding = false;
gamma_memory = 1;
threshold = 0;
propagation_rule = 'sum_enorm'; % try with sum or sum_enorm (the latter enhance results significantly but only when using tags+diseq, for all the others it's worse)
filtering_rule = {'GWsTA', 'GWsTA', 'GWsTA', 'GWsTA', 'GWsTA', 'GWsTA'}; % this is a cell array (vector of strings) because we will try several different values of c (order of cliques)
tampering_type = 'noise';

residual_memory = 0;
concurrent_cliques = 2;
no_concurrent_overlap = false;
concurrent_successive = false;
concurrent_disequilibrium = [true, 3, false, 3, true, false];
filtering_rule_first_iteration = false;
filtering_rule_last_iteration = false;

% Overlays
enable_overlays = true;
overlays_max = [0 0 0 1 1 1];
overlays_interpolation = 'uniform';

% Plot tweaking
statstries = 3; % retry n times with different networks to average (and thus smooth) the results. This will redo the whole learning phase + testing phases, so the different statstries are all statistically different (contrary to the "tests" variable while will only redo multiple test phases, not the learning phase).
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

plot_theo = true; % plot theoretical error rates?
silent = true; % If you don't want to see the progress output
save_results = true; % save results to a file?

% == Launching the runs
D = zeros(numel(M), numel(filtering_rule));
E = zeros(numel(M), numel(filtering_rule));
TE = zeros(numel(M), 1); % theoretical error rate depends on: Chi, l, c, erasures, enable_guiding and of course the density (theoretical or real) and thus on any parameter that changes the network (thus as the number of messages m to learn)
ED = zeros(numel(M), numel(filtering_rule));
SM = zeros(numel(M), numel(filtering_rule));
MM = zeros(numel(M), numel(filtering_rule));
EC = zeros(numel(M), numel(filtering_rule));

runs_total = statstries * numel(M) * numel(filtering_rule); % total number of runs we will have to do to finish this experiment (allows to compute the ETA)
tperftotal = cputime(); % total time elapsed until now (allows to compute the ETA)
rcounter = 0; % current iteration counter (allows to compute the ETA)
for t=1:statstries
    tperf = cputime(); % to show the total time elapsed later
    cnetwork = logical(sparse([]));
    thriftymessages = logical(sparse([]));
    for m=1:numel(M) % and for each value of m, we will do a run
        % Launch the run
        if m == 1
            [cnetwork, thriftymessages, density] = gbnn_learn('m', round(M(1, 1)*Mcoeff), 'miterator', miterator(1,m), 'l', l, 'c', c, 'Chi', Chi, ...
                                                                                                        'enable_overlays', enable_overlays, ...
                                                                                                        'silent', silent);
        else % Optimization trick: instead of relearning the whole network, we will reuse the previous network and just add more messages, this allows to decrease the learning time exponentially, rendering it constant (at each learning, the network will learn the same amount of messages: eg: iteration 1 will learn 1E5 messages, iteration 2 will learn 1E5 messages and reuse 1E5, which will totalize as 2E5, etc...)
            [cnetwork, s2, density] = gbnn_learn('cnetwork', cnetwork, ...
                                                        'm', round((M(1, m)-M(1,m-1))*Mcoeff), 'miterator', miterator(1,m), 'l', l, 'c', c, 'Chi', Chi, ...
                                                        'enable_overlays', enable_overlays, ...
                                                        'silent', silent);
            thriftymessages = [thriftymessages ; s2]; % append new messages
        end

        counter = 1;
        for f=1:numel(filtering_rule)
            if strcmpi(filtering_rule{f}, 'ML') % for ML it's useless to do multiple iterations
                iterations_bak = iterations;
                iterations = 1;
            end

            [error_rate, theoretical_error_rate, test_stats] = gbnn_test('cnetwork', cnetwork, 'thriftymessagestest', thriftymessages, ...
                                                                                  'erasures', erasures, 'iterations', iterations, 'tampered_messages_per_test', tampered_messages_per_test, 'tests', tests, ...
                                                                                  'enable_guiding', enable_guiding, 'gamma_memory', gamma_memory, 'threshold', threshold, 'propagation_rule', propagation_rule, 'filtering_rule', filtering_rule{f}, 'tampering_type', tampering_type, ...
                                                                                  'residual_memory', residual_memory, 'concurrent_cliques', concurrent_cliques, 'no_concurrent_overlap', no_concurrent_overlap, 'concurrent_successive', concurrent_successive, 'filtering_rule_first_iteration', filtering_rule_first_iteration, 'filtering_rule_last_iteration', filtering_rule_last_iteration, ...
                                                                                  'concurrent_disequilibrium', concurrent_disequilibrium(f), ...
                                                                                  'enable_overlays', enable_overlays, 'overlays_max', overlays_max(f), 'overlays_interpolation', overlays_interpolation, ...
                                                                                  'silent', silent);

            if strcmpi(filtering_rule{f}, 'ML') % restore the number of iterations for other filtering rules after ML
                iterations = iterations_bak;
            end

            % Store the results
            D(m,counter) = D(m,counter) + density;
            E(m,counter) = E(m,counter) + error_rate;
            TE(m) = theoretical_error_rate;
            ED(m, counter) = ED(m, counter) + test_stats.error_distance;
            SM(m, counter) = SM(m, counter) + test_stats.similarity_measure;
            MM(m, counter) = MM(m, counter) + test_stats.matching_measure;
            EC(m, counter) = EC(m, counter) + test_stats.concurrent_unbiased_error_rate;
            if ~silent; fprintf('-----------------------------\n\n'); aux.flushout(); end;

            % Update counters (for plotting and ETA)
            counter = counter + 1; % for plots
            rcounter = rcounter + 1; % for ETA

            % Display ETA
            aux.printeta(rcounter, runs_total, tperftotal, silent);
        end
    end
    aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do all runs of one statstries: %G seconds.\n'); aux.flushout(); % print total time elapsed
end
aux.printcputime(cputime() - tperftotal, 'Total cpu time elapsed to do all runs of all statstries: %G seconds.\n'); aux.flushout(); % print total time elapsed
% Normalizing errors rates by calculating the mean error for all tries
D = D ./ statstries;
E = E ./ statstries;
ED = ED ./ statstries;
SM = SM ./ statstries;
MM = MM ./ statstries;
EC = EC ./ statstries;
fprintf('END of all tests!\n'); aux.flushout();


% == Plotting

% -- First interpolate data points to get smoother curves
% Note: if smooth_factor == 1 then these commands won't change the data points nor add more.
nsamples = numel(M);
if smooth_factor > 1
    M_interp = interp1(1:nsamples, M, linspace(1, nsamples, nsamples*smooth_factor), smooth_method);
    D_interp = interp1(1:nsamples, D(:,1), linspace(1, nsamples, nsamples*smooth_factor), smooth_method);
    E_interp = interp1(D(:,1), E, D_interp, smooth_method);
    TE_interp = interp1(D(:,1), TE, D_interp, smooth_method);
    ED_interp = interp1(D(:,1), ED, D_interp, smooth_method);
    SM_interp = interp1(D(:,1), SM, D_interp, smooth_method);
    MM_interp = interp1(D(:,1), MM, D_interp, smooth_method);
    EC_interp = interp1(D(:,1), EC, D_interp, smooth_method);
else
    M_interp = M;
    D_interp = D(:,1);
    E_interp = E;
    TE_interp = TE;
    ED_interp = ED;
    SM_interp = SM;
    MM_interp = MM;
    EC_interp = EC;
end

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

% -- Plot error rate with respect to the density (or number of messages stored) and a few other parameters
figure; hold on;
xlabel(sprintf('(Bottom) Density -- (Top) Number of stored messages (M) x%.1E', Mcoeff));
ylabel(sprintf('Retrieval Error Rate for concurrent cliques=%i', concurrent_cliques));
counter = 1; % useful to keep track inside the matrix E or for plotting (to change the marker for each curve). This is guaranteed to be OK since we use the same order of for loops (so be careful, if you move the forloops here in plotting you must also move them the same way in the tests above!)
for f=1:numel(filtering_rule) % for each different filtering rule and whether there is guiding or not, we willl print a different curve, with an automatically selected color and shape
    coloridx = mod(f-1, numel(colorvec))+1; % change color per filtering rule
    lstyleidx = mod(counter-1, numel(linestylevec))+1; % change line style ...
    mstyleidx = mod(counter-1, numel(markerstylevec))+1; % and change marker style per plot

    lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
    cur_plot = plot(D_interp, E_interp(:,f), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line
    set(cur_plot, plot_curves_params{:}); % additional plot style

    fr = filtering_rule(1,f); fr = fr{1};
    plot_title = sprintf('%s', fr);
    if enable_guiding
        plot_title = strcat(plot_title, sprintf(' (guided)'));
    else
        plot_title = strcat(plot_title, sprintf(' (blind)'));
    end
    if concurrent_disequilibrium(f)
        plot_title = strcat(plot_title, sprintf(' + Diseq'));
        if concurrent_disequilibrium(f) > 1
            plot_title = strcat(plot_title, sprintf(' type %i', concurrent_disequilibrium(f)));
        end
    end
    if overlays_max(f) == 0
        plot_title = strcat(plot_title, sprintf(' + M tags'));
    elseif overlays_max(f) > 1
        plot_title = strcat(plot_title, sprintf(' + %i tags', overlays_max(f)));
    end
    set(cur_plot, 'DisplayName', plot_title); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/

    counter = counter + 1; % don't remove or the plotting markers won't be different for each curve
end

% Plot theoretical error rates
if plot_theo
    counter = counter + 1;
    coloridx = mod(counter, numel(colorvec))+1;

    lstyleidx = mod(counter-1, numel(linestylevec))+1;
    mstyleidx = mod(counter-1, numel(markerstylevec))+1;

    lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
    cur_plot = plot(D_interp, TE_interp, sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line
    set(cur_plot, plot_curves_params{:}); % additional plot style

    plot_title = '';
    plot_title = strcat(plot_title, 'Theo.');
    if enable_guiding
        plot_title = strcat(plot_title, sprintf(' (guided)'));
    else
        plot_title = strcat(plot_title, sprintf(' (blind)'));
    end
    set(cur_plot, 'DisplayName', plot_title); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
end

% Refresh plot with legends
legend(get(gca,'children'),get(get(gca,'children'),'DisplayName'), 'location', 'southeast'); % IMPORTANT: force refreshing to show the legend, else it won't show!
legend('boxoff');
% Add secondary axis on the top of the figure to show the number of messages
aux.add_2nd_xaxis(D(:,1), M, sprintf('x%.1E', Mcoeff), '%g', 0);
xlim([0 round(max(D(:,1)))]); % adjust x axis zoom
% Adjust axis drawing style
set( gca(), plot_axis_params{:} );
% Adjust text style
set([gca; findall(gca, 'Type','text')], plot_text_params{:});


% -- Plot concurrent unbiased error rate with respect to the density (or number of messages stored) and a few other parameters
figure; hold on;
xlabel(sprintf('(Bottom) Density -- (Top) Number of stored messages (M) x%.1E', Mcoeff));
ylabel(sprintf('Retrieval Error Rate (concurrent unbiased) for concurrent cliques=%i', concurrent_cliques));
counter = 1; % useful to keep track inside the matrix E or for plotting (to change the marker for each curve). This is guaranteed to be OK since we use the same order of for loops (so be careful, if you move the forloops here in plotting you must also move them the same way in the tests above!)
for f=1:numel(filtering_rule) % for each different filtering rule and whether there is guiding or not, we willl print a different curve, with an automatically selected color and shape
    coloridx = mod(f-1, numel(colorvec))+1; % change color per filtering rule
    lstyleidx = mod(counter-1, numel(linestylevec))+1; % change line style ...
    mstyleidx = mod(counter-1, numel(markerstylevec))+1; % and change marker style per plot

    lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...

    cur_plot = plot(D_interp, EC_interp(:,f), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line
    set(cur_plot, plot_curves_params{:}); % additional plot style

    fr = filtering_rule(1,f); fr = fr{1};
    plot_title = sprintf('%s', fr);
    if enable_guiding
        plot_title = strcat(plot_title, sprintf(' (guided)'));
    else
        plot_title = strcat(plot_title, sprintf(' (blind)'));
    end
    if concurrent_disequilibrium(f)
        plot_title = strcat(plot_title, sprintf(' + Diseq'));
        if concurrent_disequilibrium(f) > 1
            plot_title = strcat(plot_title, sprintf(' type %i', concurrent_disequilibrium(f)));
        end
    end
    if overlays_max(f) == 0
        plot_title = strcat(plot_title, sprintf(' + M tags'));
    elseif overlays_max(f) > 1
        plot_title = strcat(plot_title, sprintf(' + %i tags', overlays_max(f)));
    end
    set(cur_plot, 'DisplayName', plot_title); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/

    counter = counter + 1;
end

% Plot theoretical error rates
if plot_theo
    counter = counter + 1;
    coloridx = mod(counter, numel(colorvec))+1;

    lstyleidx = mod(counter-1, numel(linestylevec))+1;
    mstyleidx = mod(counter-1, numel(markerstylevec))+1;

    lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
    cur_plot = plot(D_interp, TE_interp, sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line
    set(cur_plot, plot_curves_params{:}); % additional plot style

    plot_title = '';
    plot_title = strcat(plot_title, 'Theo.');
    if enable_guiding
        plot_title = strcat(plot_title, sprintf(' (guided)'));
    else
        plot_title = strcat(plot_title, sprintf(' (blind)'));
    end
    set(cur_plot, 'DisplayName', plot_title); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
end

% Refresh plot with legends
legend(get(gca,'children'),get(get(gca,'children'),'DisplayName'), 'location', 'southeast'); % IMPORTANT: force refreshing to show the legend, else it won't show!
legend('boxoff');
% Add secondary axis on the top of the figure to show the number of messages
aux.add_2nd_xaxis(D(:,1), M, sprintf('x%.1E', Mcoeff), '%g', 0);
xlim([0 round(max(D(:,1)))]); % adjust x axis zoom
% Adjust axis drawing style
set( gca(), plot_axis_params{:} );
% Adjust text style
set([gca; findall(gca, 'Type','text')], plot_text_params{:});



% -- Plot matching_measure and other stats evolution for diseq + tags
figure; hold on;
xlabel(sprintf('(Bottom) Density -- (Top) Number of stored messages (M) x%.1E', Mcoeff));
ylim([0 1]);
f = 1; % set here the filtering rule that use disequilibrium + tags
counter = 1; % useful to keep track inside the matrix E or for plotting (to change the marker for each curve). This is guaranteed to be OK since we use the same order of for loops (so be careful, if you move the forloops here in plotting you must also move them the same way in the tests above!)

fr = filtering_rule(1,f); fr = fr{1};
if concurrent_disequilibrium(f); fr = strcat(fr, ' diseq'); end;

coloridx = mod(counter-1, numel(colorvec))+1; lstyleidx = mod(counter-1, numel(linestylevec))+1; mstyleidx = mod(counter-1, numel(markerstylevec))+1; lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
cur_plot = plot(D_interp, 1-MM_interp(:,f), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx)));
set(cur_plot, plot_curves_params{:}); % additional plot style
set(cur_plot, 'DisplayName', strcat(fr, ' - mismatching measure')); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
counter = counter + 1;

coloridx = mod(counter-1, numel(colorvec))+1; lstyleidx = mod(counter-1, numel(linestylevec))+1; mstyleidx = mod(counter-1, numel(markerstylevec))+1; lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
cur_plot = plot(D_interp, 1-SM_interp(:,f), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx)));
set(cur_plot, plot_curves_params{:}); % additional plot style
set(cur_plot, 'DisplayName', strcat(fr, ' - dissimilarity measure')); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
counter = counter + 1;

coloridx = mod(counter-1, numel(colorvec))+1; lstyleidx = mod(counter-1, numel(linestylevec))+1; mstyleidx = mod(counter-1, numel(markerstylevec))+1; lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
cur_plot = plot(D_interp, ED_interp(:,f), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx)));
set(cur_plot, plot_curves_params{:}); % additional plot style
set(cur_plot, 'DisplayName', strcat(fr, ' - error distance')); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
counter = counter + 1;

coloridx = mod(counter-1, numel(colorvec))+1; lstyleidx = mod(counter-1, numel(linestylevec))+1; mstyleidx = mod(counter-1, numel(markerstylevec))+1; lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
cur_plot = plot(D_interp, E_interp(:,f), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx)));
set(cur_plot, plot_curves_params{:}); % additional plot style
set(cur_plot, 'DisplayName', strcat(fr, ' - error rate')); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
counter = counter + 1;

% Refresh plot with legends
legend(get(gca,'children'),get(get(gca,'children'),'DisplayName'), 'location', 'northwest'); % IMPORTANT: force refreshing to show the legend, else it won't show!
% Add secondary axis on the top of the figure to show the number of messages
aux.add_2nd_xaxis(D(:,1), M, sprintf('x%.1E', Mcoeff), '%g', 0);
xlim([0 max(D(:,1))]); % adjust x axis zoom
% Adjust axis drawing style
set( gca(), plot_axis_params{:} );
% Adjust text style
set([gca; findall(gca, 'Type','text')], plot_text_params{:});


% Print densities values and error rates
fprintf('Densities:\n'); disp(D);
fprintf('Error rates:\n'); disp(E);
fprintf('Theoretical error rates:\n'); disp(TE);

% The end!
