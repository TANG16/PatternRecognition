function i = getIndex(distance, metric_types)
    i = find(cellfun(@(x)strcmp(x, distance),metric_types));
end

