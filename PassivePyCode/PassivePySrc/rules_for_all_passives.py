import spacy
from spacy.matcher import Matcher

def create_matcher(spacy_model = "en_core_web_lg", nlp:spacy.language.Language = None):

    """
    creates a matcher with a SpaCy nlp model. The default model is:
    https://spacy.io/models/en#en_core_web_lg
    
    """
    if not nlp:
        nlp = spacy.load(spacy_model, disable=["ner"])
    matcher = Matcher(nlp.vocab)

    # list of verbs that their adjective form 
    # is sometimes mistaken as a verb
    verbs_list = ["associate", "involve", "exhaust", "base", 
                "lead", "stun", "overrate",  "fill", "bear",
                "complicate", "reserve", "complicate", "heat",
                "screw",]

    #--------------------------rules--------------------#

    
    passive_rule_1 = [
        {"POS":"AUX", "DEP": "aux", "OP":"*"},
        {"POS":"AUX", "DEP": "auxpass", "OP":"+"},
        {"DEP":"neg", "TAG":"RB", "OP":"*"},
        {"DEP":"HYPH", "OP":"*"},
        {"DEP":"advmod", "TAG":"RB", "OP":"*"},
        {"POS":"VERB", "TAG":"VBN", "LEMMA":{"NOT_IN" : verbs_list + ['be']}}
    ]

    """
    sentence : The book was read by him.
    dependencies : ['det', 'nsubjpass', 'auxpass', 'ROOT', 'agent', 'pobj', 'punct']
    Tags : ['DT', 'NN', 'VBD', 'VBN', 'IN', 'PRP', '.']
    """
    passive_rule_2 = [
        {"DEP": {"IN": ["attr", 'nsubjpass', 'appos']}},
        {"TAG": "RB", "DEP": "advmod", "OP" : "*"},
        {"DEP": "PUNCT", "OP" : "*"},
        {"TAG": "VBN", "DEP": "acl","LEMMA": {"NOT_IN" : verbs_list}}
    ]

    """
    sentence : there was no change detected in her behavior.
    dependencies : ['expl', 'ROOT', 'det', 'attr', 'acl', 'prep', 'poss', 'pobj', 'punct']
    tags : ['EX', 'VBD', 'DT', 'NN', 'VBN', 'IN', 'PRP$', 'NN', '.']
    """


    passive_rule_3 = [
        {"POS":"AUX", "DEP": "aux", "OP":"*"},
        {"POS":"AUX", "DEP": "auxpass", "OP":"+"},
        {"DEP":"neg", "TAG":"RB", "OP":"*"},
        {"DEP":"HYPH", "OP":"*"},
        {"DEP":"advmod", "TAG":"RB", "OP":"*"},
        {"POS":"VERB", "DEP":"ROOT", "LEMMA":{"NOT_IN" : verbs_list}},
        {"DEP":"cc"},
        {"DEP":"advmod", "TAG":"VBN", "OP": "*", "LEMMA": {"NOT_IN":["pre"]}},
        {"DEP": "conj", "LEMMA":{"NOT_IN" : verbs_list}},
        {"DEP":"pobj", "OP":"!"}
    ]

    """
    Used for the second part with "and ..." 
    sentence : it was determined and formed.
    dependencies : ['nsubjpass', 'auxpass', 'ROOT', 'cc', 'conj', 'punct']
    tags : ['PRP', 'VBD', 'VBN', 'CC', 'VBN', '.']
    """

    passive_rule_4 = [
        {"DEP":"advcl", "TAG":"VBN"},
        {"DEP": "agent", "TAG":"IN"},
        {"OP":"*"},
        {"DEP": "pobj"},
    ]

    """
    sentence : killed by the police, he never thought this would be his end.
    dependencies : ['advcl', 'agent', 'det', 'pobj', 'punct', 'nsubj', 'neg', 'ROOT', 'nsubj', 'aux', 'ccomp', 'poss', 'attr']
    tags : ['VBN', 'IN', 'DT', 'NN', ',', 'PRP', 'RB', 'VBD', 'DT', 'MD', 'VB', 'PRP$', 'NN']
    """


    passive_rule_5 = [
        {"LEMMA": {"IN": verbs_list}},
        {"LOWER":"by"}
    ]


    """
    to avoid the confusion between the adjective and passive version of specific 
    verbs, we dedicated a new rule to some verbs to be detected when used with 
    an agent (by)

    sentence : Natural resources are exhausted by humans.
    dependencies : ['amod', 'nsubjpass', 'auxpass', 'ROOT', 'agent', 'pobj']
    tags : ['JJ', 'NNS', 'VBP', 'VBN', 'IN', 'NNS']
    """


    # ------------------adding rules to the matcher----------#

    matcher.add("passive_rule_1", [passive_rule_1], greedy='LONGEST')
    matcher.add("passive_rule_2", [passive_rule_2], greedy='LONGEST')
    matcher.add("passive_rule_3", [passive_rule_3], greedy='LONGEST')
    matcher.add("passive_rule_4", [passive_rule_4], greedy='LONGEST')
    matcher.add("passive_rule_5", [passive_rule_5], greedy='LONGEST')

    # print('Matcher is built.')

    return nlp, matcher