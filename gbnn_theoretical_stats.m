function cnetwork_stats = gbnn_theoretical_stats(varargin)
% cnetwork_stats = gbnn_theoretical_stats(varargin)
%
% Compute some theoretical stats and predictions (like the optimum number of messages to store) for the Gripon-Berrou Neural Network (sparse cluster version from 2014 Behrooz paper).
%
% Use variable arguments, example:
% gbnn_theoretical_stats('Chi', 100, 'l', 64, 'c', 16, 'd', 0.33);
% 
%
% Chi = number of clusters
% c = order of a clique = number of fanals in a clique (can be a scalar or a vector of min and max c)
% l = number of fanals per cluster
% d = density = ratio between currently existing edges over the total number of possible edges (graph_size)
%
% Note: works only on cliques networks, not on tournaments.
% Most equations are from Behrooz 2014 paper: Storing Sparse Messages in Networks of Neural Cliques

% == Importing some useful functions
% source('gbnn_aux.m'); % does not work with MatLab, only Octave...
aux = gbnn_aux; % works with both MatLab and Octave

% == Arguments processing
% List of possible arguments and their default values
arguments_defaults = struct( ...
    ... % Mandatory
    'Chi', 0, ...
    'c', 0, ...
    'l', 0, ...
    ...
    ... % One is mandatory but others are optional
    'M', 0, ...
    'd', 0, ...
    'false_positive_rate', 0, ...
    ...
    ... % Totally optional
    'erasures', -1, ...
    'error_rate', 0, ...
    ...
    ... % Debug stuffs
    'silent', false);

% Process the arguments
arguments = aux.getnargs(varargin, arguments_defaults, true);

% Load variables into local namespace (called workspace in MatLab)
aux.varspull(arguments);


% == Do some work

% Sanity checks
if erasures > 0 && erasures < 1
    alpha = erasures;
    erasures = c * alpha;
end

% Go on with stats
cnetwork_stats = struct();
n = Chi*l; % total number of fanals (nodes) in the network
% Fixed length messages case
if numel(c) == 1
    % Code encoding perspective: is the code really redundant and with a high merit rate?
    clique_size = ((c-1) .* c) / 2; % size of a clique = number of edges in a clique (message)
    min_distance = 2 .* (c-1); % minimum distance between two messages/cliques (number of edges to erase and add to exchange one fanal/character of the message/clique with another)
	data_per_clique = floor((c + 1) / 2); % Nombre minimal d'arêtes nécessaires pour spécifier complètement une clique à c sommets (on sait qu'il existe c sommets juste en regardant ces arêtes)
	redundancy_number = (clique_size - data_per_clique); % = r = codeword = number of redundant nodes, that is the surplus of nodes that are not necessary to define the clique (but are useful to increase the minimum distance).
	redundancy_rate = redundancy_number / data_per_clique; % or tau
	Fmerit = min_distance / (1 + redundancy_rate); % merit rate of the code encoder
	autofix = floor((min_distance - 1) / 2) % = c - 2 in this case

    % Graph density (the basis of the computations)
    graph_size = (Chi*(Chi-1) * l.^2) / 2; % graph size = maximum number of potential connexions that may exist in the graph, which is equivalent to the binary resource Q (so the unit is either the number of edges, or in bits).
    if d > 0; theoretical_density = d; end;
    if M > 0
        theoretical_density = 1 - (1 - (c*(c-1)) / (Chi*(Chi-1) * l.^2))^M; % theoretical density if we are given the number of messages to store
        if d == 0; d = theoretical_density; end;
    end
    if false_positive_rate > 0
        theoretical_density = 2^(log2(false_positive_rate)/clique_size); % another way to compute the theoretical density if we are given the false positive rate
        if d == 0; d = theoretical_density; end;
    end

    % Compute all the other stats
    M = log2(1 - d) / log2(1 - (c*(c-1)) / (Chi*(Chi-1) * l.^2)); % maximum number of messages (cliques) that can be stored with this network (and density). This is the exact formula, derived from the theoretical_density = 1 - (1 - (c*(c-1)) / (Chi*(Chi-1) * l.^2))^M
    Mapprox = ((Chi*(Chi-1) * l.^2) / c*(c-1)) * d ; % approximation of the maximum number of messages that can be stored (this approximation is valid only when M << l^2)
    entropy_per_message = (log2(nchoosek(Chi, c)) + (c * log2(l))); % entropy_per_message = binary resource per message = shannon information contained in each message
    entropy = M * entropy_per_message; % total entropy for all messages
    efficiency = entropy / graph_size; % efficaciency = eta = B/Q = entropy/graph_size = ratio between amount of information storable in the network over the resource/material necessary to use to store this amount of information.
    Mmax = graph_size / entropy_per_message; % Efficiency-1 = optimal number of messages to store in the network to get the best efficiency.
    false_positive_rate = d ^ clique_size; % error rate of second type (false positive)
    if erasures >= 0 || error_rate > 0
        if error_rate > 0
            theoretical_error_rate = error_rate;
        else
            theoretical_error_rate = 1 - (1 - d^(c-erasures))^(erasures*(l - 1) + (Chi - c)*l); % theoretical error rate
            error_rate = theoretical_error_rate;
        end
        c_optimal_approx = log((Chi * l) / error_rate) / (2 * (1 - (erasures/c))); % approximation of the optimal length/order of messages/cliques in order to optimize the diversity (number of messages stored) given a target error rate
    end
% Variable-length case
elseif numel(c) == 2
    clique_size_min = ((min(c)-1) .* min(c)) / 2;
    clique_size_max = ((max(c)-1) .* max(c)) / 2;

    lambda = max(c) - min(c) + 1;
    graph_size = (Chi*(Chi-1) * l.^2 * lambda) / 2;
    density_per_c = @(c) (1 - (c*(c-1)) / (Chi*(Chi-1)*l.^2)).^(M*c);
    if d > 0; theoretical_density = d; end;
    if M > 0
        theoretical_density = 1 - prod(arrayfun(@(c) density_per_c(c), c)); % TODO: derive this equation to get M if we only know the density.
        if d == 0; d = theoretical_density; end;
    end

    entropy_per_message = @(c) (log2(nchoosek(Chi, c)) + (c * log2(l)));
    entropy_all_orders = sum(arrayfun(@(c) entropy_per_message(c), c));
    if M > 0
        entropy = M * entropy_all_orders;
        efficiency = entropy / graph_size;
    else
        entropy = NaN;
        efficiency = NaN;
    end
    Mmax = graph_size / entropy_all_orders;
    if erasures >= 0
        error_rate_per_c = @(c) (1 - (1 - d^(c-erasures))^(erasures*(l-1) + (Chi-c)*l));
        theoretical_error_rate = 1 / lambda * sum(arrayfun(@(c) error_rate_per_c(c), c));
    end

end

% Pack them all and return the stats (we do this separately so that calculus equations are not bloated by object-oriented syntax)
cnetwork_stats.graph_size = graph_size;
cnetwork_stats.theoretical_density = theoretical_density;
cnetwork_stats.entropy_per_message = entropy_per_message;
cnetwork_stats.entropy = entropy;
cnetwork_stats.efficiency = efficiency;
cnetwork_stats.Mmax = Mmax;
if erasures >= 0 || error_rate > 0
    cnetwork_stats.theoretical_error_rate = theoretical_error_rate;
end
% Add specific stats for special cases
if numel(c) == 1
    cnetwork_stats.clique_size = clique_size;
    cnetwork_stats.min_distance = min_distance;
    cnetwork_stats.data_per_clique = data_per_clique;
    cnetwork_stats.redundancy_number = redundancy_number;
    cnetwork_stats.redundancy_rate = redundancy_rate;
    cnetwork_stats.Fmerit = Fmerit;
    cnetwork_stats.autofix = autofix;
    
    cnetwork_stats.M = M;
    cnetwork_stats.Mapprox = Mapprox;
    cnetwork_stats.false_positive_rate = false_positive_rate;
    if erasures >= 0 || error_rate > 0; cnetwork_stats.c_optimal_approx = c_optimal_approx; end;
end
if numel(c) == 2
    cnetwork_stats.clique_size_min = clique_size_min;
    cnetwork_stats.clique_size_max = clique_size_max;
    cnetwork_stats.lambda = lambda;
    cnetwork_stats.density_per_c = density_per_c;
    cnetwork_stats.entropy_all_orders = entropy_all_orders;
    if erasures >= 0
        cnetwork_stats.error_rate_per_c = error_rate_per_c;
    end
end


end % endfunction
