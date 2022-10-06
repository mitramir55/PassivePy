import spacy
from spacy.matcher import Matcher


def create_matcher(nlp:spacy.language.Language = None, spacy_model = "en_core_web_lg"):

    """creates a matcher on the following vocabulary"""
    if not nlp:
        if spacy_model == "en_core_web_lg":
            import en_core_web_lg
            nlp = en_core_web_lg.load(disable=["ner"])
        elif spacy_model == "en_core_web_md":
            import en_core_web_md
            nlp = en_core_web_md.load(disable=["ner"])
        elif spacy_model == "en_core_web_sm":
            import en_core_web_sm
            nlp = en_core_web_sm.load(disable=["ner"])
        else:
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
        {"POS":"VERB", "TAG":"VBN", "LEMMA":{"NOT_IN" : verbs_list + ['be']}},
        {"DEP":"agent", "OP":"!"}
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
        {"TAG": "VBN", "DEP": "acl","LEMMA": {"NOT_IN" : verbs_list}},
        {"DEP":"agent", "OP":"!"}
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
        {"DEP":"agent", "OP":"!"}
    ]

    """
    Used for the second part with "and ..." 
    sentence : it was determined and formed.
    dependencies : ['nsubjpass', 'auxpass', 'ROOT', 'cc', 'conj', 'punct']
    tags : ['PRP', 'VBD', 'VBN', 'CC', 'VBN', '.']
    """


    passive_rule_5 = [
        {"DEP": "nsubj"},
        {"DEP": "ROOT"},
        {"DEP": "attr", "TAG": "VBN", "LEMMA":{"NOT_IN" : verbs_list}},
        {"DEP": "prep", "TAG": "IN", "OP":"*"},
        {"DEP":"agent", "OP":"!"}
    ]

    """
    sentence : Bears are dreamt of in your fantasies!
    dependencies : ['nsubjpass', 'auxpass', 'ROOT', 'prep', 'prep', 'poss', 'pobj', 'punct']
    tags : ['NNS', 'VBP', 'VBN', 'IN', 'IN', 'PRP$', 'NNS', '.']
    """



    # ------------------adding rules to the matcher----------#

    matcher.add("passive_rule_1", [passive_rule_1], greedy='LONGEST')
    matcher.add("passive_rule_2", [passive_rule_2], greedy='LONGEST')
    matcher.add("passive_rule_3", [passive_rule_3], greedy='LONGEST')
    matcher.add("passive_rule_5", [passive_rule_5], greedy='LONGEST')

    # print('Matcher is built.')

    return matcher
