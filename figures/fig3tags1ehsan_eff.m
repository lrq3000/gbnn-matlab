% Overlays network: Behrooz vs overlays benchmark. Please use Octave >= 3.8.1 for reasonable performances!
% To print a figure: print(1, 'test.eps', '-color');
% On Windows, use IrfanView and GhostScript 32-bits to read .eps files without quality loss (all the others will show a deformed image).

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
set (0, 'defaultaxesfontname', 'Helvetica'); % fix GhostScript error about handling unknown fonts
graphics_toolkit('gnuplot'); % fix truncated curves in output file

% Preparing stuff to automate the plots
% This will allow us to automatically select a different color and shape for each curve
colorvec = 'rgbmc';
markerstylevec = '+o*.xsd^v><ph';
linestylevec = {'-' ; ':' ; '--' ; '-.'};
linestylestd = ':';

% Vars config, tweak the stuff here
%M = [0.1 0.5:0.1:1 1.25:0.25:2 2.5 3:2:7 11 16 25 40]; % this is a vector because we will try several values of m (number of messages, which influences the density)
M = [0.02180727066532258 0.1044843119959677 0.1237871723790323 0.1427750126008064 0.161345451108871 0.1795221144153226 0.1974703881048387 0.240691154233871 0.2810846144153226 0.3199896043346774 0.3560909148185484 0.4231035786290323 0.4834850680443548 0.6676616053427419 0.7855027721774194 0.9107941658266129 0.9701518397177419 0.9960149949596774 0.9998346144153226];
%M = [0.02180727066532258 0.1044843119959677 0.1237871723790323 0.1427750126008064 0.1974703881048387 0.240691154233871 0.2810846144153226 0.4231035786290323 0.4834850680443548]; % just to quickly try the std plotting
%M = [0.01 0.1:0.1:0.9 0.95];
%M = [0.005 5.1]; % to test both limits to check that the range is OK, the first point must be near 0 and the second point must be near 1, at least for one of the curves
%Mcoeff = 1E3;
Mcoeff = 1;
miterator = zeros(1,numel(M)); %M/2;
c = 8;
l = 16;
Chi = 32;
erasures = floor(c/2); %floor(c*0.25);
iterations = 1; % for convergence
tampered_messages_per_test = 200;
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
overlays_max = [1 0 20];
overlays_interpolation = {'uniform'};
enable_overlays_guiding = false;

% Plot tweaking
statstries = 5; % retry n times with different networks to average (and thus smooth) the results
smooth_factor = 3; % interpolate more points to get smoother curves. Set to 1 to avoid smoothing (and thus plot only the point of the real samples).
smooth_method = 'cubic'; % use PCHIP or cubic to avoid interpolating into negative values as spline does
plot_curves_params = { 'markersize', 7, ...
                                            'linewidth', 3 ...
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
D = zeros(numel(statstries), numel(M), numel(overlays_max)*numel(overlays_interpolation));
E = zeros(numel(statstries), numel(M), numel(overlays_max)*numel(overlays_interpolation));
EFF = zeros(numel(statstries), numel(M), numel(overlays_max)*numel(overlays_interpolation));
TE = zeros(numel(M), numel(overlays_max)); % theoretical error rate depends on: Chi, l, c, erasures, enable_guiding and of course the density (theoretical or real) and thus on any parameter that changes the network (thus as the number of messages m to learn)

for t=1:statstries
    tperf = cputime(); % to show the total time elapsed later
    cnetwork = logical(sparse([]));
    thriftymessages = logical(sparse([]));
    for m=1:numel(M) % and for each value of m, we will do a run
        % Launch the run
        if m == 1
            if M(1) * Mcoeff < 1
                nbmes = M(1);
            else
                nbmes = round(M(1)*Mcoeff);
            end
            [cnetwork, thriftymessages, density] = gbnn_learn('m', nbmes, 'miterator', miterator(1,m), 'l', l, 'c', c, 'Chi', Chi, ...
                                                                                                        'enable_overlays', enable_overlays, ...
                                                                                                        'silent', silent);
        else % Optimization trick: instead of relearning the whole network, we will reuse the previous network and just add more messages, this allows to decrease the learning time exponentially, rendering it constant (at each learning, the network will learn the same amount of messages: eg: iteration 1 will learn 1E5 messages, iteration 2 will learn 1E5 messages and reuse 1E5, which will totalize as 2E5, etc...)
            if M(1) * Mcoeff < 1 % if density is specified, we have to convert to the number of messages and then compute the number of messages we have to learn to add up over previously learned messages. This is the only accurate way of computing the number of messages to learn with online learning, else if we do density-density_prev, the number of messages won't be correct (eg: number of messages at density 0.2 and the number of messages to learn between 0.6 and 0.8 is vastly different, in the latter case we have a lot more messages to learn than just the number to reach 0.2 density).
                cnetwork_stats = gbnn_theoretical_stats('Chi', Chi, 'c', c, 'l', l, 'd', M(m));
                cnetwork_stats_prev = gbnn_theoretical_stats('Chi', Chi, 'c', c, 'l', l, 'd', M(m-1));
                nbmes = round(cnetwork_stats.M - cnetwork_stats_prev.M);
                clear cnetwork_stats;
            else
                nbmes = round((M(m)-M(m-1))*Mcoeff);
            end
            [cnetwork, s2, density] = gbnn_learn('cnetwork', cnetwork, ...
                                                        'm', nbmes, 'miterator', miterator(1,m), 'l', l, 'c', c, 'Chi', Chi, ...
                                                        'enable_overlays', enable_overlays, ...
                                                        'silent', silent);
            thriftymessages = [thriftymessages ; s2]; % append new messages
        end

        counter = 1;
        for om=1:numel(overlays_max)
            for oi=1:numel(overlays_interpolation)
                [error_rate, theoretical_error_rate] = gbnn_test('cnetwork', cnetwork, 'thriftymessagestest', thriftymessages, ...
                                                                                      'erasures', erasures, 'iterations', iterations, 'tampered_messages_per_test', tampered_messages_per_test, 'tests', tests, ...
                                                                                      'enable_guiding', enable_guiding, 'gamma_memory', gamma_memory, 'threshold', threshold, 'propagation_rule', propagation_rule, 'filtering_rule', filtering_rule, 'tampering_type', tampering_type, ...
                                                                                      'residual_memory', residual_memory, 'filtering_rule_first_iteration', filtering_rule_first_iteration, 'filtering_rule_last_iteration', filtering_rule_last_iteration, ...
                                                                                      'enable_overlays', enable_overlays, 'overlays_max', overlays_max(om), 'overlays_interpolation', overlays_interpolation{oi}, 'enable_overlays_guiding', enable_overlays_guiding, ...
                                                                                      'silent', silent);

                if M(1) * Mcoeff < 1
                    cnetwork_stats = gbnn_theoretical_stats('Chi', Chi, 'c', c, 'l', l, 'd', M(m), 'erasures', erasures, 'overlays_max', overlays_max(om));
                else
                    cnetwork_stats = gbnn_theoretical_stats('Chi', Chi, 'c', c, 'l', l, 'M', M(m)*Mcoeff, 'erasures', erasures, 'overlays_max', overlays_max(om));
                end

                % Store the results
                %colidx = counter+(size(D,2)/numel(enable_overlays))*(o-1);
                D(t, m,counter) = density;
                E(t, m,counter) = error_rate;
                EFF(t, m,counter) = cnetwork_stats.efficiency;
                TE(m, om) = theoretical_error_rate;

                clear cnetwork_stats;

                if ~silent; fprintf('-----------------------------\n\n'); end;

                counter = counter + 1;
            end
        end
    end
    aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do all runs: %G seconds.\n'); aux.flushout(); % print total time elapsed
end
fprintf('END of all tests!\n'); aux.flushout();

% Backup the full data (they will be saved to a file below)
D_full = D;
E_full = E;
EFF_full = EFF;
% Normalizing errors rates by calculating the mean error for all tries
D = squeeze(mean(D_full));
E = squeeze(mean(E_full));
EFF = squeeze(mean(EFF_full));
% Compute standard deviation
D_std = squeeze(std(D_full));
E_std = squeeze(std(E_full));
EFF_std = squeeze(std(EFF_full));

% Print densities values and error rates
%fprintf('Densities:\n'); disp(D);
%fprintf('Error rates:\n'); disp(E);
%fprintf('Efficiencies:\n'); disp(EFF);
%fprintf('Theoretical error rates:\n'); disp(TE);
%aux.flushout();

% == Plotting

% -- First interpolate data points to get smoother curves
% Note: if smooth_factor == 1 then these commands won't change the data points nor add more.
nsamples = numel(M);
if smooth_factor > 1
    M_interp = interp1(1:nsamples, M, linspace(1, nsamples, nsamples*smooth_factor), smooth_method);
    D_interp = interp1(1:nsamples, D(:,1), linspace(1, nsamples, nsamples*smooth_factor), smooth_method);
    E_interp = interp1(D(:,1), E, D_interp, smooth_method);
    EFF_interp = interp1(D(:,1), EFF, D_interp, smooth_method);
    TE_interp = interp1(D(:,1), TE, D_interp, smooth_method);
else
    M_interp = M;
    D_interp = D;
    E_interp = E;
    EFF_interp = EFF;
    TE_interp = TE;
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

% -- Plot ratio efficiency/error rate with respect to the density (or number of messages stored) and a few other parameters
figure; hold on;
xlabel(sprintf('(Bottom) Density  -- (Top) Number of stored messages (M) x%.1E', Mcoeff));
ylabel('Ratio efficiency / real error rate');
counter = 1; % useful to keep track inside the matrix E. This is guaranteed to be OK since we use the same order of for loops (so be careful, if you move the forloops here in plotting you must also move them the same way in the tests above!)
for om=numel(overlays_max):-1:1
    for oi=1:numel(overlays_interpolation)
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
        plot_title2 = strcat(plot_title, ' - error rate');

        set(cur_plot, 'DisplayName', plot_title2); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/


        counter = counter + 1;
    end
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
xlim([0 round(max(D(:,1))*10)/10]); % adjust x axis zoom
ylim([0 1]);
% Adjust axis drawing style
set( gca(), plot_axis_params{:} );
% Adjust text style
set([gca; findall(gca, 'Type','text')], plot_text_params{:});
% Add grid
grid on;



% -- Plot efficiency with respect to error rate
figure; hold on;
xlabel('Error rate');
ylabel('Efficiency');
counter = 1; % useful to keep track inside the matrix E. This is guaranteed to be OK since we use the same order of for loops (so be careful, if you move the forloops here in plotting you must also move them the same way in the tests above!)
for om=1:numel(overlays_max)
    for oi=1:numel(overlays_interpolation)
        colorcounter = om;
        if numel(overlays_interpolation) > 1; colorcounter = oi; end;
        coloridx = mod(colorcounter-1, numel(colorvec))+1; % change color if overlay or not
        
        % -- Set title
        plot_title = sprintf('%s', filtering_rule);
        if enable_guiding
            plot_title = strcat(plot_title, ' - Guided');
        else
            plot_title = strcat(plot_title, ' - Blind');
        end
        if overlays_max(om) == 1
            plot_title = strcat(plot_title, ' - One/No tags');
        elseif overlays_max(om) == 0
            plot_title = strcat(plot_title, ' - M tags');
        else
            plot_title = strcat(plot_title, sprintf(' - %i tags', overlays_max(om)));
        end
        plot_title2 = plot_title; % back up plot title for the theo curve
        plot_title = strcat(plot_title, sprintf(' - %i it', iterations));

        % -- Efficiency 1
        lstyleidx = mod(counter-1, numel(linestylevec))+1; % change line style ...
        mstyleidx = mod(counter-1, numel(markerstylevec))+1; % and change marker style per plot

        lstyle = linestylevec(1, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...

        cur_plot = plot(E_interp(:,counter), EFF_interp(:,counter), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line
        set(cur_plot, plot_curves_params{:}); % additional plot style

        set(cur_plot, 'DisplayName', plot_title); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
        
        % -- Standard deviation for efficiency
        try % plot std of error rate because eff has no deviation since it's computed theoretically (and is close enough to the real efficiency if the number of really learnt messages is close enough to the number of messages we intended to learn).
            std_plot1 = plot(E(:,counter) + E_std(:,counter), EFF(:,counter), sprintf('%s%s', linestylestd, colorvec(coloridx)));
            std_plot2 = plot(E(:,counter) - E_std(:,counter), EFF(:,counter), sprintf('%s%s', linestylestd, colorvec(coloridx)));
            set(std_plot1, 'HandleVisibility', 'off');
            set(std_plot2, 'HandleVisibility', 'off');
            set(std_plot1, plot_curves_params{:}); % additional plot style
            set(std_plot2, plot_curves_params{:}); % additional plot style
        catch err
            warning('Could not plot the standard deviation!');
            warning(err);
        end

        % -- Theoretical efficiency
        lstyle = linestylevec(3, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
        cur_plot = plot(TE_interp(:,counter), EFF_interp(:,counter), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line
        set(cur_plot, plot_curves_params{:}); % additional plot style
        plot_title2 = strcat(plot_title2, ' - theo');

        set(cur_plot, 'DisplayName', plot_title2); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/


        counter = counter + 1;
    end
end


% Plot theoretical error rates
if plot_theo
    %coloridx = mod(counter, numel(colorvec))+1;
    colornm = 'k';
    counter = 1;
    for om=1:numel(overlays_max)
        lstyleidx = mod(counter-1, numel(linestylevec))+1;
        mstyleidx = mod(counter-1, numel(markerstylevec))+1;

        lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
        cur_plot = plot(D_interp, TE_interp(:,counter), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colornm)); % plot one line
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
legend(get(gca,'children'),get(get(gca,'children'),'DisplayName'), 'location', 'southeast'); % IMPORTANT: force refreshing to show the legend, else it won't show!
legend('boxoff');
legend('left'); % Bug workaround: as of Octave 3.8.1, gnuplot produce weird legend text, with a huge blank space because it horizontally align legend text to the right, and there's no way currently to change to left. Only solution is to move the text to the left and symbols to the right, this way there's no blank space anymore.
% Setup axis
xlim([0 round(max(max(E)))]); % adjust x axis zoom
ylim([0 1]);
% Adjust axis drawing style
set( gca(), plot_axis_params{:} );
% Adjust text style
set([gca; findall(gca, 'Type','text')], plot_text_params{:});
% Add grid
grid on;

% The end!
