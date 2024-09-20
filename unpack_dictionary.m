function value = unpack_dictionary(dictionary_path, key)

    dictionary = load(dictionary_path);
    dictionary = dictionary.dP_dict;

    % Unpacks a dictionary with a given key. Unpacking method is depending on the method argument.
    value = dictionary(key);
    value = value{1};

end