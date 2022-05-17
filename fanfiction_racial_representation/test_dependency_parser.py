import os
import stanza
from stanza.server import CoreNLPClient
from graphviz import Source

NUM_FILE_CHECKPOINT = 10


# NLTK is deprecating stanford_dependency_parser. Use  nltk.parse.corenlp.CoreNLPDependencyParser instead.
# https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK#english
def setup_nltk_stanford_dependency_parser():
    from nltk.parse.stanford import StanfordDependencyParser
    # make sure nltk can find stanford-parser
    # please check your stanford-parser version from brew output (in my case 3.6.0)

    if os.name == 'nt':
        os.environ['JAVA_HOME'] = r'D:\Program Files\Java\jre1.8.0_271'
        os.environ['CLASSPATH'] = r'D:\usr\local\Cellar\stanford-parser\4.2.0\libexec'
    else:
        os.environ['CLASSPATH'] = r'/usr/local/Cellar/stanford-parser/4.2.0/libexec'
    # TODO: try StanfordNeuralDependencyParser
    sdp = StanfordDependencyParser()
    # example:
    # sentence = 'The brown fox is quick and he is jumping over the lazy dog'
    # result = list(sdp.raw_parse(sentence))
    # dep_tree_dot_repr = [parse for parse in result][0].to_dot()
    # source = Source(dep_tree_dot_repr, filename="dep_tree", format="png")
    # source.view()

    return sdp


def setup_allennlp_coreference_predictor():
    from tensorflow.keras.layers import MaxPooling2D, Conv2D, Input, Add, MaxPool2D, Flatten, AveragePooling2D, Dense, \
        BatchNormalization, ZeroPadding2D, Activation, Concatenate, UpSampling2D
    from tensorflow.keras.models import Model

    from allennlp.predictors.predictor import Predictor
    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")
    return predictor


def setup_stanza_corenlp_dependency_parser():
    # https://stanfordnlp.github.io/stanza/depparse.html
    stanza.download('en')
    if os.name == 'nt':
        nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', use_gpu=True)
    else:
        nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse')
    return nlp


def setup_spacy_neuralcoref():
    # python -m spacy download en
    # python -m spacy download en_core_web_lg
    import spacy
    import neuralcoref

    nlp = spacy.load('en_core_web_lg')
    nlp.max_length = 4000000
    neuralcoref.add_to_pipe(nlp)
    return nlp


def name_replace_preprocess_doc_raw(doc_raw):
    replacement_dict = {'Shang - Chi': 'Shang-Chi', 'Shang Chi': 'Shang-Chi', 'Ant Man': 'Ant-Man',
                        'Ant - Man': 'Ant-Man', 'Yon - Rogg': 'Yon-Rogg', 'Yon Rogg': 'Yon-Rogg',
                        'Spider - Man': 'Spider-Man', 'Spider Man': 'Spider-Man', 'Star Lord': 'Star-Lord',
                        'Star Lord': 'Star-Lord'}
    for key in replacement_dict.keys():
        doc_raw = doc_raw.replace(key, replacement_dict[key])
    return doc_raw


def setup_deprecated_stanford_corenlp():
    from stanfordcorenlp import \
        StanfordCoreNLP  # used for coreference resolution https://github.com/Lynten/stanford-corenlp/wiki/Coreference-Resolution
    if os.name == 'nt':
        stanford_corenlp = StanfordCoreNLP(r'D:\stanford-corenlp-4.3.2')
        return stanford_corenlp
    else:
        assert (0 and "path not specified yet")
    # example:
    # stanford_corenlp = setup_stanford_corenlp()
    # text = "My sister has a friend called John. Really, tell me more about him? She think he is so funny!"
    # print(stanford_corenlp.coref(text))
    # stanford_corenlp.close()


def is_coref_cluster_in_character_set(coref_cluster, character_to_identifier_dict):
    for mention in coref_cluster.mentions:
        if mention.lemma_.strip() in character_to_identifier_dict:
            return character_to_identifier_dict[mention.lemma_.strip()], (mention.lemma_.strip(), mention.end_char)
        for first_last_name in mention.lemma_.strip().split():
            if first_last_name in character_to_identifier_dict:
                return character_to_identifier_dict[first_last_name], (first_last_name, mention.end_char)
    return -1


def my_generic_binary_search_(end_char, container, comparator_func, l, r):
    if r >= l:
        mid = (r - l) // 2 + l
        if comparator_func(end_char, container[mid]) == 0:
            # end_char is found in container[mid]
            return mid
        if comparator_func(end_char, container[mid]) == 1:
            # end_char is before container[mid]
            # search in [l,mid-1]
            return my_generic_binary_search_(end_char, container, comparator_func, l, mid - 1)
        # end_char is after container[mid]
        # search in [mid+1,r]
        return my_generic_binary_search_(end_char, container, comparator_func, mid + 1, r)
    return -1


def my_generic_binary_search(end_char, container, comparator_func):
    # binary search code is from https://www.geeksforgeeks.org/binary-search/
    l = 0
    r = len(container)
    return my_generic_binary_search_(end_char, container, comparator_func, l, r)


def locate_end_char_in_stanza_document_annotation(end_char, stanza_document_annotation):
    def sentence_end_char_comparator(end_char, sentence):
        if sentence.words[0].end_char > end_char:
            return 1  # current sentence index is larger then the one's to find
        elif sentence.words[-1].end_char < end_char:
            return -1  # current sentence index is smaller then the one's to find
        else:
            return 0  # current sentence index is of the one to find

    def word_end_char_comparator(end_char, word):
        if word.end_char > end_char:
            return 1  # current word index is larger then the one's to find
        elif word.end_char < end_char:
            return -1  # current word index is smaller then the one's to
        else:
            return 0  # current word index is of the one to find

    sentence_idx = my_generic_binary_search(end_char, stanza_document_annotation.sentences,
                                            sentence_end_char_comparator)
    word_idx = my_generic_binary_search(end_char, stanza_document_annotation.sentences[sentence_idx].words,
                                        word_end_char_comparator)
    return sentence_idx, word_idx


import collections

FoundAdjective = collections.namedtuple("FoundAdjective", ["adjective", "index_of_the_found_word_refer_to_the_name",
                                                           "the_found_word_refer_to_the_name", "sentence"])
AdjectivesForACorefCluster = collections.namedtuple("AdjectivesForACorefCluster",
                                                    ["filename", "sentence_where_name_is_found",
                                                     "the_found_word_refer_to_the_name",
                                                     "index_of_the_found_word_refer_to_the_name", "adjectives"])


def find_adjective_for_noun_in_sentence(stanza_sentence, noun_word_idx):
    adjectives = []
    # First, dealing with case 1,2,3
    # case 1: I am actually drained. {I} is {nsubj:pass} for {drained}. {drained} is a VERB. Dole was defeated by Clinton is the same case.
    # case 2: The baby is cute. {baby} is {nsubj} for {cute}. {cute} is a ADJ.
    # case 3: Sam is cool and smart. {Sam} is {nsub} for {cool}. {cool} is a ADJ. {smart} is {conj} for {cool}. {smart} is {conj}.
    noun_word = stanza_sentence.words[noun_word_idx]
    if noun_word.deprel == "nsubj:pass" or (
            noun_word.deprel == "nsubj" and noun_word.head > 0 and stanza_sentence.words[
        noun_word.head - 1].pos == "ADJ"):
        # more about pos tagging: https://web.stanford.edu/~jurafsky/slp3/slides/8_POSNER_intro_May_6_2021.pdf
        # Only extract text from the word annotation because that is what we care about
        adjective = stanza_sentence.words[noun_word.head - 1].text if noun_word.head > 0 else "root"
        # This is case 1. Add this to result
        adjectives.append(
            FoundAdjective(adjective=adjective, index_of_the_found_word_refer_to_the_name=noun_word_idx,
                           the_found_word_refer_to_the_name=noun_word.text, sentence=stanza_sentence.text))
        for word in stanza_sentence.words:
            if word.deprel == "conj" and word.head == noun_word.head and noun_word.head > 0:
                conj_adjective = word.text
                # This is case 3. Add this to result
                adjectives.append(
                    FoundAdjective(adjective=conj_adjective, index_of_the_found_word_refer_to_the_name=noun_word_idx,
                                   the_found_word_refer_to_the_name=noun_word.text,
                                   sentence=stanza_sentence.text))  # extract text

    # Now dealing with case 4: Sam eats red meat. {red} is {amod} for {meat}.
    for word in stanza_sentence.words:
        if word.deprel == "amod" and word.head - 1 == noun_word_idx:
            amod_adjective = word.text
            # This is case 4. Add this to result
            adjectives.append(
                FoundAdjective(adjective=amod_adjective, index_of_the_found_word_refer_to_the_name=noun_word_idx,
                               the_found_word_refer_to_the_name=noun_word.text, sentence=stanza_sentence.text))
    return adjectives


def find_adjective_for_cluster(coref_cluster, stanza_document_annotation):
    result_adjectives = []
    for mention in coref_cluster.mentions:
        sentence_idx, word_idx = locate_end_char_in_stanza_document_annotation(mention.end_char,
                                                                               stanza_document_annotation)
        curr_adjectives = find_adjective_for_noun_in_sentence(stanza_document_annotation.sentences[sentence_idx],
                                                              word_idx)
        result_adjectives = result_adjectives + curr_adjectives
    return result_adjectives


def save_to_pickle_file(var_to_save, pickle_filename):
    import pickle
    pickle_out = open(pickle_filename, "wb")
    pickle.dump(var_to_save, pickle_out)
    pickle_out.close()


def load_ignore_list(previous_log_filename, log_filename):
    ignore_filelist = set()
    num_lines = 0
    with open(previous_log_filename) as fd:
        for line in fd:
            if line.find("skipping") != 0 and line.find("working on") != 0:
                num_lines += 1
    idx_line = 0
    with open(log_filename, 'a') as lfd:
        with open(previous_log_filename) as fd:
            for line in fd:
                if line.find("skipping") != 0 and line.find("working on") != 0:
                    idx_line += 1
                    if (idx_line - 1) >= (num_lines // NUM_FILE_CHECKPOINT) * NUM_FILE_CHECKPOINT:
                        lfd.write("skipping {filename} from checkpoint\n".format(filename=line.strip()[5:]))
                        continue
                    ignore_filelist.add(line.strip()[5:])
    return ignore_filelist


def find_character_adjectives_in_file(complete_filename, identifier_to_adjective_dict, character_to_identifier_dict,
                                      coref_nlp, nlp):
    with open(complete_filename) as fd:
        doc_raw = name_replace_preprocess_doc_raw(fd.read())
        doc = nlp(doc_raw)
        # ann = client.annotate(doc_raw)
        coref_annotation = coref_nlp(doc_raw)
        # coref_annotation._.coref_clusters is a list of Clusters
        # each cluster has the `mentions` member variable, which is a list of Span
        # you can use the `end_char` to align it with stanza_nlp output
        for coref_cluster in coref_annotation._.coref_clusters:
            identifier = is_coref_cluster_in_character_set(coref_cluster, character_to_identifier_dict)

            if identifier != -1:  # this coref cluster is in the character_to_identifier_dict
                identifier, (found_first_last_name, found_first_last_name_end_char) = identifier
                found_first_last_name_statement_idx, found_first_last_name_word_idx = locate_end_char_in_stanza_document_annotation(
                    found_first_last_name_end_char, doc)
                adjectives = find_adjective_for_cluster(coref_cluster, doc)
                if len(adjectives) == 0:
                    continue
                if identifier not in identifier_to_adjective_dict:
                    identifier_to_adjective_dict[identifier] = []
                identifier_to_adjective_dict[identifier] += [
                    AdjectivesForACorefCluster(filename=complete_filename,
                                               sentence_where_name_is_found=doc.sentences[
                                                   found_first_last_name_statement_idx].text,
                                               the_found_word_refer_to_the_name=
                                               doc.sentences[found_first_last_name_statement_idx].words[
                                                   found_first_last_name_word_idx].text,
                                               index_of_the_found_word_refer_to_the_name=found_first_last_name_word_idx,
                                               adjectives=adjectives)]


if __name__ == "__main__":
    FOLDER_NAME = './output_panther'
    LOG_FILENAME = "test_dependency_parser.log"
    # PREV_LOG_FILENAME_LIST = ["test_dependency_parser.log.927"]
    PREV_LOG_FILENAME_LIST = []
    # sdp = setup_nltk_stanford_dependency_parser()
    # for root, directories, files in os.walk(FOLDER_NAME, topdown=False):
    #     for name in files:
    #         complete_filename = os.path.join(root, name)
    #         with open(complete_filename) as fd:
    #             for line in fd:
    #                 if len(line.strip()) == 0:
    #                     continue
    #                 result = list(sdp.raw_parse(line))
    #                 dep_tree_dot_repr = [parse for parse in result][0].to_dot()
    #                 source = Source(dep_tree_dot_repr, filename="dep_tree", format="png")
    #                 source.view()

    # stanza.install_corenlp()
    # with CoreNLPClient(properties='en',memory='1G') as client:

    # set up the result dictionary and load character dictionary
    ignore_filelist = set()
    for PREV_LOG_FILENAME in PREV_LOG_FILENAME_LIST:
        curr_ignore_filelist = load_ignore_list(PREV_LOG_FILENAME, LOG_FILENAME)
        ignore_filelist = ignore_filelist.union(curr_ignore_filelist)
    identifier_to_adjective_dict = dict()
    import pickle
    from create_panther_character_list_to_identifier_dict import load_firstlast_name_to_identifier_dict_form_txt

    character_to_identifier_dict = load_firstlast_name_to_identifier_dict_form_txt(
        "firstlast_name_to_identifier_mapping.txt")

    # set up neuralcoref and stanza annotator
    coref_nlp = setup_spacy_neuralcoref()
    nlp = setup_stanza_corenlp_dependency_parser()

    # walk through every article text file
    for root, directories, files in os.walk(FOLDER_NAME, topdown=False):
        idx_file = 0
        for name in files:
            complete_filename = os.path.join(root, name)
            if complete_filename in ignore_filelist:
                with open(LOG_FILENAME, 'a') as lfd:
                    lfd.write("skipping {filename}\n".format(filename=complete_filename))
                continue
            # if complete_filename in {'./output_panther/19240606.txt'}:
            #    with open(LOG_FILENAME, 'a') as lfd:
            #        lfd.write("skipping {filename} due to too large size\n".format(filename=complete_filename))
            #    continue
            idx_file += 1
            with open(LOG_FILENAME, 'a') as lfd:
                lfd.write("working on {filename}\n".format(filename=complete_filename))
            find_character_adjectives_in_file(complete_filename, identifier_to_adjective_dict,
                                              character_to_identifier_dict, coref_nlp, nlp)
            with open(LOG_FILENAME, 'a') as lfd:
                lfd.write("done {filename}\n".format(filename=complete_filename))
            if (idx_file - 1) % NUM_FILE_CHECKPOINT == 0:
                save_to_pickle_file(identifier_to_adjective_dict, "identifier_to_adjective_dict.pickle")

        # ignoring directories
    import pickle

    save_to_pickle_file(identifier_to_adjective_dict, "identifier_to_adjective_dict.pickle")
    pass
