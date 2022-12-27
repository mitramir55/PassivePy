
#  Copyright Mitra Mirshafiee, 2022 Licensed under MIT License.
#  See the LICENSE.txt for more information.

import spacy
from spacy.matcher import Matcher


def create_matcher_truncated(nlp:spacy.language.Language = None):

    """creates a matcher on the following vocabulary"""

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

    # ------------------adding rules to the matcher----------#

    matcher.add("passive_rule_1", [passive_rule_1], greedy='LONGEST')
    matcher.add("passive_rule_2", [passive_rule_2], greedy='LONGEST')
    matcher.add("passive_rule_3", [passive_rule_3], greedy='LONGEST')

    # print('Matcher is built.')

    return matcher
