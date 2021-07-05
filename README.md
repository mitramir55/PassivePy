# PassivePy: A Tool to Automatically Identify Passive Voice in Big Text Data


Our aim with this work is to create a reliable (e.g., passive voice judgments are consistent), valid (e.g., passive voice judgments are accurate), flexible (e.g., texts can be assessed at different units of analysis), replicable (e.g., the approach can be performed by a range of research teams with varying levels of computational expertise), and scalable way (e.g., small and large collections of texts can be analyzed) to capture passive voice from different corpora for social and psychological evaluations of text. To achieve these aims, we introduce PassivePy, a fully transparent and documented Python library.

For using the package, first we have to install the requirements:
```
!pip install -r https://raw.githubusercontent.com/mitramir55/PassivePy/main/PassivePyCode/PassivePySrc/requirements.txt
!pip install PassivePy==0.1.0

```
Then, import PassivePy and create the matcher:

```
from PassivePySrc import PassivePy

passivepy = PassivePy.PassivePyAnalyzer()
matcher = passivepy.create_matcher()
```

And at last, we have many options such as sentence- or corpus-level analysis which requires a dataset with at least one column in any format (e.g., CSV, XLSX or XLS). 

``` 
# sentence level:
df_detected_s = passivepy.match_sentence_level(matcher, df=df, colName = 'name_of_column')

# corpus level
df_detected_c = passivepy.match_corpus_level(matcher, df=df, colName = 'name_of_column')
```


