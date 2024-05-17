function value = unpack_dictionary(dictionary_path, key, method)

    dictionary = load(dictionary_path);
    dictionary = dictionary.dP_dict;

    % Unpacks a dictionary with a given key. Unpacking method is depending on the method argument.
    PA_TO_MMHG = 0.00750061683;

    value = dictionary(key);
    value = value{1};

    if isequal(method, "vWERP")
        value = value.dP .* PA_TO_MMHG;
    end

end