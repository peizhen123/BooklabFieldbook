import pickle

if __name__=="__main__":
    pickle_in = open("character_name_to_character_dict.pickle","rb")
    character_name_to_charcter_dict = pickle.load(pickle_in)
    for