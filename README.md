# PassivePy: A Tool to Automatically Identify Passive Voice in Big Text Data


Our aim with this work is to create a reliable (e.g., passive voice judgments are consistent), valid (e.g., passive voice judgments are accurate), flexible (e.g., texts can be assessed at different units of analysis), replicable (e.g., the approach can be performed by a range of research teams with varying levels of computational expertise), and scalable way (e.g., small and large collections of texts can be analyzed) to capture passive voice from different corpora for social and psychological evaluations of text. To achieve these aims, we introduce PassivePy, a fully transparent and documented Python library.

For using the package, first we have to install the requirements:
```
!pip install -r https://raw.githubusercontent.com/mitramir55/PassivePy/main/PassivePyCode/PassivePySrc/requirements_lg.txt
!pip install -i https://test.pypi.org/simple/ PassivePy==0.0.37

```
Then, import PassivePy and initiate the analyzer:

```
from PassivePySrc import PassivePy

spacy_model = "en_core_web_lg"
passivepy = PassivePy.PassivePyAnalyzer(spacy_model)
```

And at last, we have can either analyze the records in sentence- or corpus-level. Your dataset can be in any format (e.g., CSV, XLSX or XLS).; however, make sure to that it has at least one column with the text that needs to be analyzed.

In case of large datasets, you can also add `batch_size = ...` and `n_process=...` to speed up the analysis (the default for both is 1).


``` 
# sentence level:
df_detected_s = passivepy.match_sentence_level(df=df, colName = 'abstract_text')

# corpus level
df_detected_c = passivepy.match_corpus_level(df=df, colName = 'abstract_text')
```

If you do not need any columns to be appended to the main dataset, simply add `add_other_columns = False`, or if you don't what the percentages to show up add `percentage_of_passive_sentences = False` in any of the following functions.


