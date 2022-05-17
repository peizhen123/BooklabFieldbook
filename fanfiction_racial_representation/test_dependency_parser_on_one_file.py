from test_dependency_parser import *

if __name__ == "__main__":
    coref_nlp = setup_spacy_neuralcoref()
    nlp = setup_stanza_corenlp_dependency_parser()
    complete_filename = './output_panther/19240606.txt'
    import pickle

    from create_panther_character_list_to_identifier_dict import load_firstlast_name_to_identifier_dict_form_txt

    character_to_identifier_dict = load_firstlast_name_to_identifier_dict_form_txt(
        "firstlast_name_to_identifier_mapping.txt")

    identifier_to_adjective_dict = dict()

    find_character_adjectives_in_file(complete_filename, identifier_to_adjective_dict,
                                      character_to_identifier_dict, coref_nlp, nlp)

    save_to_pickle_file(identifier_to_adjective_dict, "19240606_identifier_to_adjective_dict.pickle")
