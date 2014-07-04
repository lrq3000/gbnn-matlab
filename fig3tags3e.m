% Overlays network: Willshaw vs overlays benchmark. Please use Octave >= 3.8.1 for reasonable performances!

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
linestylevec = {'-' ; '--' ; ':' ; '-.'};

% Vars config, tweak the stuff here
M = 0.005:1:5.1; % this is a vector because we will try several values of m (number of messages, which influences the density)
%M = [0.005 5.1]; % to test both limits to check that the range is OK, the first point must be near 0 and the second point must be near 1, at least for one of the curves
Mcoeff = 10E2;
miterator = zeros(1,numel(M)); %M/2;
c = 8;
l = 1;
Chi = 256;
erasures = floor(c*0.25);
iterations = 2; % for convergence
tampered_messages_per_test = 30;
tests = 1;

enable_guiding = false;
gamma_memory = 0;
threshold = 0;
filtering_rule = 'GWsTA';
propagation_rule = 'overlays_filter';
tampering_type = 'erase';

residual_memory = 0;
GWTA_first_iteration = false;
GWTA_last_iteration = false;

% Overlays
enable_overlays = true;
%overlays_max = [1 2 5 20 100 0];
overlays_max = [1 5 0];
overlays_interpolation = {'mod'};

statstries = 2;

silent = false; % If you don't want to see the progress output
thnodraw = true; % draw theoretical error rates?

% == Launching the runs
D = zeros(numel(M), numel(overlays_max)*numel(overlays_interpolation));
E = zeros(numel(M), numel(overlays_max)*numel(overlays_interpolation));
TE = zeros(numel(M), numel(overlays_max)); % theoretical error rate depends on: Chi, l, c, erasures, enable_guiding and of course the density (theoretical or real) and thus on any parameter that changes the network (thus as the number of messages m to learn)

for t=1:statstries
    tperf = cputime(); % to show the total time elapsed later
    cnetwork = logical(sparse([]));
    thriftymessages = logical(sparse([]));
    for m=1:numel(M) % and for each value of m, we will do a run
        % Launch the run
        if m == 1
            [cnetwork, thriftymessages, density] = gbnn_learn('m', round(M(1, 1)*Mcoeff), 'miterator', miterator(1,m), 'l', l, 'c', c, 'Chi', Chi, ...
                                                                                                        'enable_overlays', enable_overlays, 'overlays_max', overlays_max, 'overlays_interpolation', overlays_interpolation, ...
                                                                                                        'silent', silent);
        else % Optimization trick: instead of relearning the whole network, we will reuse the previous network and just add more messages, this allows to decrease the learning time exponentially, rendering it constant (at each learning, the network will learn the same amount of messages: eg: iteration 1 will learn 1E5 messages, iteration 2 will learn 1E5 messages and reuse 1E5, which will totalize as 2E5, etc...)
            [cnetwork, s2, density] = gbnn_learn('cnetwork', cnetwork, ...
                                                        'm', round((M(1, m)-M(1,m-1))*Mcoeff), 'miterator', miterator(1,m), 'l', l, 'c', c, 'Chi', Chi, ...
                                                        'enable_overlays', enable_overlays, 'overlays_max', overlays_max, 'overlays_interpolation', overlays_interpolation, ...
                                                        'silent', silent);
            thriftymessages = [thriftymessages ; s2]; % append new messages
        end

        counter = 1;
        for om=1:numel(overlays_max)
            cnetwork.primary.args.overlays_max = overlays_max(om);
            for oi=1:numel(overlays_interpolation)
                cnetwork.primary.args.overlays_interpolation = overlays_interpolation(oi);

                prop_rule = 'sum';
                if (enable_overlays && overlays_max(om) ~= 1); prop_rule = propagation_rule; end;
                if (overlays_max(om) == 1); temp = cnetwork; cnetwork.primary.net = logical(cnetwork.primary.net); end;
                [error_rate, theoretical_error_rate] = gbnn_test('cnetwork', cnetwork, 'thriftymessagestest', thriftymessages, ...
                                                                                      'erasures', erasures, 'iterations', iterations, 'tampered_messages_per_test', tampered_messages_per_test, 'tests', tests, ...
                                                                                      'enable_guiding', enable_guiding, 'gamma_memory', gamma_memory, 'threshold', threshold, 'propagation_rule', prop_rule, 'filtering_rule', filtering_rule, 'tampering_type', tampering_type, ...
                                                                                      'residual_memory', residual_memory, 'GWTA_first_iteration', GWTA_first_iteration, 'GWTA_last_iteration', GWTA_last_iteration, ...
                                                                                      'silent', silent);
                if (overlays_max(om) == 1); cnetwork = temp; end;

                % Store the results
                %colidx = counter+(size(D,2)/numel(enable_overlays))*(o-1);
                D(m,counter) = D(m,counter) + density;
                E(m,counter) = E(m,counter) + error_rate;
                TE(m, om) = theoretical_error_rate;
                if ~silent; fprintf('-----------------------------\n\n'); end;

                counter = counter + 1;
            end
        end
    end
    aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do all runs: %G seconds.\n'); aux.flushout(); % print total time elapsed
end
% Normalizing errors rates by calculating the mean error for all tries
D = D ./ statstries;
E = E ./ statstries;
printf('END of all tests!\n'); aux.flushout();


% == Plotting

% Print densities values and error rates
fprintf('Densities:\n'); disp(D);
fprintf('Error rates:\n'); disp(E);
fprintf('Theoretical error rates:\n'); disp(TE);
aux.flushout();

% Plot error rate with respect to the density (or number of messages stored) and a few other parameters
figure; hold on;
xlabel(sprintf('Number of stored messages (M) x %.1E', Mcoeff));
ylabel('Retrieval Error Rate');
counter = 1; % useful to keep track inside the matrix E. This is guaranteed to be OK since we use the same order of for loops (so be careful, if you move the forloops here in plotting you must also move them the same way in the tests above!)
for om=numel(overlays_max):-1:1
    for oi=1:numel(overlays_interpolation)
        colorcounter = om;
        if numel(overlays_interpolation) > 1; colorcounter = oi; end;
        coloridx = mod(colorcounter-1, numel(colorvec))+1; % change color if overlay or not

        lstyleidx = mod(counter-1, numel(linestylevec))+1; % change line style ...
        mstyleidx = mod(counter-1, numel(markerstylevec))+1; % and change marker style per plot

        lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
        cur_plot = plot(M, E(:,numel(overlays_max)*numel(overlays_interpolation)+1-counter), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line

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
            plot_title = strcat(plot_title, sprintf(' (%s)', overlays_interpolation{oi}));
        end
        set(cur_plot, 'DisplayName', plot_title); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/

        counter = counter + 1;
    end
end


% Plot theoretical error rates
if ~thnodraw
    %coloridx = mod(counter, numel(colorvec))+1;
    colornm = 'k';
    counter = 1;
    for om=numel(overlays_max):-1:1
        lstyleidx = mod(counter-1, numel(linestylevec))+1;
        mstyleidx = mod(counter-1, numel(markerstylevec))+1;

        lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
        cur_plot = plot(M, TE(:,om), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colornm)); % plot one line

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
legend(get(gca,'children'),get(get(gca,'children'),'DisplayName')); % IMPORTANT: force refreshing to show the legend, else it won't show!

% The end!
