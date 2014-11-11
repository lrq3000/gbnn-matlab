function [decoded_messages, maxidxs] = gbnn_dbsearch(varargin)
%
% decoded_messages, maxidxs = gbnn_dbsearch(learned_messages, decoded_messages, ...
%                                                                                  concurrent_cliques, ...
%                                                                                  silent)
%
% Exhaustive search inside the database of learned messages to find which one corresponds the most to the decoded messages (matching by similarity score).
% NOTE: the decoded_messages should use the filtering rule GWSTA-ML to maximize the probability to find the original message, because GWSTA-ML will only remove fanals that are surely not part of any clique, thus GWSTA-ML is the only filtering rule with a matching score always equal to 1 (the maximum).
%
% This function supports named arguments, use it like this:
% gbnn_dbsearch('learned_messages', testset, 'decoded_messages', partial_messages)
%
% Where testset and partial_messages are the variables returned by gbnn_test.m
% 


% == Importing some useful functions
aux = gbnn_aux; % works with both MatLab and Octave

% == Arguments processing
% List of possible arguments and their default values
arguments_defaults = struct( ...
    ... % Mandatory
    'learned_messages', [], ... % must be thrifty
    'decoded_messages', [], ... % must be thrifty
    ...
    ... % Tests tweakings and rules
    'concurrent_cliques', 1, ... % 1 is disabled, > 1 enables and specify the number of concurrent messages/cliques to decode concurrently
    ...
    ... % Debug stuffs
    'silent', false);

% Process the arguments
arguments = aux.getnargs(varargin, arguments_defaults, true);

% Load variables into local namespace (called workspace in MatLab)
aux.varspull(arguments);

% == Sanity Checks
if isempty(learned_messages) || isempty(decoded_messages)
    error('Missing arguments: learned_messages and decoded_messages are mandatory (and they must be thrifty)!');
end

n = size(decoded_messages, 1);


% == Database search

% One clique to find only
if concurrent_cliques == 1
    % Get the most similar messages from the database
    [~, maxidxs] = max(learned_messages * decoded_messages); % decoded_messages * learned_messages'  compute the similarity measure: more a message match one from the database, higher will be the score. This is like a database LIKE search: a decoded_message may have more characters than the most similar learned_message but anyway we will pick the one that share most of its characters with the decoded_message. Since both matrices are binary, it is enough to matrix multiply them, so that each decoded message is multiplied (matched) to every learned messages, and at the end we will have a similarity score per decoded message for every learned messages (thus a bidimensional matrix: score per decoded message and per learned message). This is very much like a distance matrix (manhattan or euclidian, in our binary environment it's the same).

    % Extract messages
    learned_messages = learned_messages';
    decoded_messages = learned_messages(:, maxidxs);

% For multiple concurrent cliques
elseif concurrent_cliques > 1
    % Compute the full similarity score for each decoded message to every messages in the database, and sort by order
    % Note: this will compare the whole decoded messages (containing multiple cliques) with database messages containing only one clique per message (but that's no problem since we compute a similarity score)
    [~, maxidxs] = sort(learned_messages * decoded_messages, 'descend');
    
    % Fetch as many database message as we have cliques per decoded message
    maxidxs = maxidxs(1:concurrent_cliques, :);

    % Extract the messages (one clique per message)
    learned_messages = learned_messages';
    decoded_messages = learned_messages(:, maxidxs'); % we use maxidxs'(:) to prepare for reshape, so that instead of having DM = [A1 B1 ; A2 B2] and thus DM(:) = [A1 A2 B1 B2] we have DM' = [A1 A2 ; B1 B2] and thus DM'(:) = [A1 B1 A2 B2] so that all first cliques are listed in the first row and second cliques will be reshaped to fit in the second row (to reshape we have to separate all things of first class and later put all things of second class).

    % Merge the cliques
    decoded_messages = reshape(any(reshape(decoded_messages, [], concurrent_cliques)', 1)', n, []);
end

end % endfunction
