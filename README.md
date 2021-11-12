# PassivePy: A Tool to Automatically Identify Passive Voice in Big Text Data


Our aim with this work is to create a reliable (e.g., passive voice judgments are consistent), valid (e.g., passive voice judgments are accurate), flexible (e.g., texts can be assessed at different units of analysis), replicable (e.g., the approach can be performed by a range of research teams with varying levels of computational expertise), and scalable way (e.g., small and large collections of texts can be analyzed) to capture passive voice from different corpora for social and psychological evaluations of text. To achieve these aims, we introduce PassivePy, a fully transparent and documented Python library.

For using the package, first we have to install the requirements:
```
!pip install -r https://raw.githubusercontent.com/mitramir55/PassivePy/main/PassivePyCode/PassivePySrc/requirements_lg.txt
!pip install -i https://test.pypi.org/simple/ PassivePy==0.0.56

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
passivepy.match_text(sample_text)
```
The output will be:
```
sentence : the input sentence
binary : Whether any passive voice is detected 
passive_match(es) : The span of passive form in text
raw_passive_count : Number of passive voices
```
For processing datasets, we have can either analyze the records sentence- or corpus-level. Your dataset can be in any format (e.g., CSV, XLSX or XLS).; however, make sure to that it has at least one column with the text that needs to be analyzed.

In case of large datasets, you can also add `batch_size = ...` and `n_process=...` to speed up the analysis (the default for both is 1).


``` 
# sentence level:
df_detected_s = passivepy.match_sentence_level(df=df, column_name = 'text')

# corpus level
df_detected_c = passivepy.match_corpus_level(df=df, column_name = 'text')
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

If you needed to analyze each token of a sentence, i.g., print out the `DEP` (dependency), `POS` (coarse-grained POS tags), `TAG` (fine-grained part of speech tags), `LEMMA` (canonical form) of a word,  you can use the `parse_sentence` method of passivepy in the following way:

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
