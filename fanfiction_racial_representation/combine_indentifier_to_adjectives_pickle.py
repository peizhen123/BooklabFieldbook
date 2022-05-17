import pickle
from test_dependency_parser import AdjectivesForACorefCluster, FoundAdjective, save_to_pickle_file


def combine_identifier_to_adjective_dicts(dict1, dict2):
    result = dict()
    for key in {key for key in dict1.keys()}.union({key for key in dict2.keys()}):
        result[key] = list()
        if key in dict1:
            result[key] += dict1[key]
        if key in dict2:
            result[key] += dict2[key]
    return result


if __name__ == "__main__":
    PICKLES_TO_COMBINE = ['../identifier_to_adjective_dict.pickle', '../identifier_to_adjective_dict.pickle.927', '../19240606_identifier_to_adjective_dict.pickle']
    COMBINED_PICKLE_FILENAME = 'combined_identifier_to_adjective_dict.pickle'
    combined_identifier_to_adjective_dict = dict()
    for filename in PICKLES_TO_COMBINE:
        with open(filename, 'rb') as fd:
            curr_part_character_to_identifier_dict = pickle.load(fd)
            combined_identifier_to_adjective_dict = combine_identifier_to_adjective_dicts(
                curr_part_character_to_identifier_dict, combined_identifier_to_adjective_dict)
            pass
    save_to_pickle_file(combined_identifier_to_adjective_dict, COMBINED_PICKLE_FILENAME)
    pass
