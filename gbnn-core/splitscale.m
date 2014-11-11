function y = splitscale(x, p, mode)
% Mathematical transform to compute a continuous function that is logarithmic when x is high enough (higher than p), and linear when x is lower than x
% From the poster "A Mathematically Simple Alternative to the Logarithmic Transform for Flow Cytometric Fluorescence Data Displays", 2005, Francis L. Battye
% Note: mode can either be 'splitscale' (default) or 'biexponential'
% TODO: add hyperlog

    if ~exist('mode', 'var')
        mode = 'splitscale';
    end
    
    if strcmpi(mode, 'splitscale')
        if x >= p
            y = log(x);
        else
            A = (log(p)/p);
            y = A * x;
        end
    elseif strcmpi(mode, 'biexponential') || strcmpi(mode, 'biexp')
        biexp = @(x, b) (log(x+sqrt(x.^2 + b)));
        y = biexp(x, p); % best value for b: 1, this way you get a negative exponential with the exact same asymptotic behavior than the positive one (but inversed of course)
    end
end
