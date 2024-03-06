import spacy
from spacy.matcher import Matcher

def create_matcher_german(spacy_model="de_core_news_sm", nlp: spacy.language.Language = None):
    if not nlp:
        nlp = spacy.load(spacy_model, disable=["ner"])
    matcher = Matcher(nlp.vocab)

    passiv_regel_1 = [

        {"LEMMA": {"IN": ["werden", "sein"]}, "OP": "+"},
        {"TAG": {"IN": ["ADJA", "ADJD", "ADV", "APPO", "APPR", "APPRART", "APZR", "ART", "CARD", "FM", "ITJ", "KOKOM",
                        "KON", "KOUI", "KOUS", "NE", "NN", "NNE", "PDAT", "PDS", "PIAT", "PIS", "PPER", "PPOSAT",
                        "PPOSS", "PRELAT", "PRELS", "PRF", "PROAV", "PTKA", "PTKANT", "PTKNEG", "PTKVZ", "PTKZU",
                        "PWAT", "PWAV", "PWS", "TRUNC", "VAFIN", "VAIMP", "VAINF", "VAPP", "VMFIN", "VMINF", "VMPP",
                        "VVFIN", "VVIMP", "VVINF", "VVIZU", "VVPP", "XY", "_SP"]}, "OP": "*"},
        {"POS": "VERB", "TAG": "VVPP", "OP": "+"}
    ]

    passiv_regel_2 = [

        {"POS": "VERB", "TAG": "VVPP", "OP": "+"},
        {"LEMMA": "werden", "OP": "+"}
    ]

    matcher.add("passiv_regel_1", [passiv_regel_1], greedy='LONGEST')
    matcher.add("passiv_regel_2", [passiv_regel_2], greedy='LONGEST')
    return nlp, matcher


