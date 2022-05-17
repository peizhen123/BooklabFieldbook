import pickle
from test_dependency_parser import save_to_pickle_file

INSEPERABLE_NAME_SET = {'Winter Soldier'}


def store_coarse_firstlast_name_to_identifier_mapping_to_txt(character_list_filename, coarse_txt_filename):
    with open(coarse_txt_filename, 'w') as ofd:
        with open(character_list_filename) as fd:
            for line in fd:
                line = line.strip()
                if line.find("/") != -1:
                    names = [name.strip() for name in line.split("/")]
                    character_identifier = line.strip()
                else:
                    names = [line.strip()]
                    character_identifier = line.strip()
                for name in names:
                    if name not in INSEPERABLE_NAME_SET:
                        first_or_last_names = name.split()
                        for first_or_last_name in first_or_last_names:
                            ofd.write("{name} -> {identifier}\n".format(name=first_or_last_name,
                                                                        identifier=character_identifier))
                        # TODO: handle dash such as spider-man
                    else:
                        ofd.write("{name} -> {identifier}\n".format(name=name, identifier=character_identifier))


def load_firstlast_name_to_identifier_dict_form_txt(txt_filename):
    firstlast_name_to_identifier_dict = dict()
    with open(txt_filename) as fd:
        for line in fd:
            firstlast_name, identifier = line.split("->")
            firstlast_name = firstlast_name.strip()
            identifier = identifier.strip()
            if firstlast_name in firstlast_name_to_identifier_dict:
                print("{name} already exists in identifier dict!".format(name=firstlast_name))
            firstlast_name_to_identifier_dict[firstlast_name] = identifier
    return firstlast_name_to_identifier_dict


if __name__ == "__main__":
    store_coarse_firstlast_name_to_identifier_mapping_to_txt("panther_character_list.txt",
                                                             "coarse_firstlast_name_to_identifier_mapping.txt")
    firstlast_name_to_identifier_dict = load_firstlast_name_to_identifier_dict_form_txt(
        "coarse_firstlast_name_to_identifier_mapping.txt")

    pass
