# PassivePy: A Tool to Automatically Identify Passive Voice in Big Text Data


Our aim with this work is to create a reliable (e.g., passive voice judgments are consistent), valid (e.g., passive voice judgments are accurate), flexible (e.g., texts can be assessed at different units of analysis), replicable (e.g., the approach can be performed by a range of research teams with varying levels of computational expertise), and scalable way (e.g., small and large collections of texts can be analyzed) to capture passive voice from different corpora for social and psychological evaluations of text. To achieve these aims, we introduce PassivePy, a fully transparent and documented Python library.

For accessing the datasets in our paper, please click on [this link](https://osf.io/j2b6u/?view_only=0e78d7f4028041b693d6b64547b514ca). 

If you haven't used Python before or need detailed instructions about how to use this package please visit [our website](https://mitramir55.github.io/PassivePyWeb/).


First we have to install the requirements in the following way (all requirements are needed for spaCy or other libraries we use.):
```
!pip install -r https://raw.githubusercontent.com/mitramir55/PassivePy/main/PassivePyCode/PassivePySrc/requirements_lg.txt
!pip install PassivePy==0.2.15

```
Then, import PassivePy and initiate the analyzer:

```
from PassivePySrc import PassivePy

spacy_model = "en_core_web_lg"
passivepy = PassivePy.PassivePyAnalyzer(spacy_model)
```
Use passivepy for single sentences:
```
# Try changing the sentence below:
sample_text = "The painting has been drawn."
passivepy.passivepy.match_text(sample_text, full_passive=True, truncated_passive=True)
```
The output will be:
```
sentence : the input sentence
binary : Whether any passive voice is detected 
passive_match(es) : The span of passive form in text
raw_passive_count : Number of passive voices
```
You can set the full_passive or truncated_passive to true if you want the same sort of output for these two types of passive. (truncated is a passive without an object of preposition, while a full passive is one with the object of preposition - e.g., this was broken by him.)


For processing datasets, we have can either analyze the records sentence- or corpus-level. Your dataset can be in any format (e.g., CSV, XLSX or XLS).; however, make sure to that it has at least one column with the text that needs to be analyzed.

In case of large datasets, you can also add `batch_size = ...` and `n_process=...` to speed up the analysis (the default for both is 1).


``` 
# sentence level:
df_detected_s = passivepy.match_sentence_level(df, column_name='documents', n_process = 1,
                                                batch_size = 1000, add_other_columns=True,
                                                truncated_passive=False, full_passive=False)

# corpus level
df_detected_c = passivepy.match_corpus_level(df, column_name='sentences', n_process = 1,
                                            batch_size = 1000, add_other_columns=True,
                                            truncated_passive=False, full_passive=False)
```
In the output you will have a data frame with the following columns:

```
# corpus level
document : Records in the input data frame
binary : Whether a passive was detected in that document
passive_match(es) : Parts of the document detected as passive
raw_passive_count : Number of passive voices detected in the sentence
raw_passive_sents_count : Number of sentences with passive voice
raw_sentence_count : Number of sentences detected in the document
passive_sents_percentage : Proportion of passive sentences to total number of sentences

# Sentence level
docId : Initial index of the record in the input file
sentenceId : The ith sentence in one specific record
sentence : The detected sentence
binary : Whether a passive was detected in that sentence
passive_match(es) : The part of the record detected as passive voice
raw_passive_count : Number of passive forms detected in the sentence

```

If you needed to analyze each token of a sentence, i.g., print out the `DEP` (dependency), `POS` (coarse-grained part of speech tags), `TAG` (fine-grained part of speech tags), `LEMMA` (canonical form) of a word,  you can use the `parse_sentence` method of passivepy in the following way:

```
sample_sentence = "She has been killed"
passivepy.parse_sentence(sample_sentence)
```
And the output will be like the sample below:
```
word: She 
pos: PRON 
dependency: nsubjpass 
tag:  PRP 
lemma:  she
...
```



If you do not need any columns to be appended to the main dataset, simply add `add_other_columns = False`, or if you don't what the percentages to show up add `percentage_of_passive_sentences = False` in any of the following functions.


Accuracy on the CoLA dataset: 0.97
Accuracy on the CrowdSource Dataset: 0.98

This repository is the code of the following paper, so if you use this package, please cite the work as:
Sepehri, A., Markowitz, D. M., & Mir, M. (2022, February 3). PassivePy: A Tool to Automatically Identify Passive Voice in Big Text Data. Retrieved from psyarxiv.com/bwp3t

