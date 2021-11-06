import pandas as pd
import numpy as np
import spacy
from spacy.matcher import Matcher
from termcolor import colored
import time
import regex as re
from itertools import chain 
import string
from tqdm import tqdm 
import tqdm.notebook as tq
import os, sys


try: 
    from PassivePyCode.PassivePySrc.PassivePyRules_FullPassive import create_matcher_full
    from PassivePyCode.PassivePySrc.PassivePyRules_TruncatedPassive import create_matcher_truncated
except: 
     
    from PassivePySrc.PassivePyRules_FullPassive import create_matcher_full
    from PassivePySrc.PassivePyRules_TruncatedPassive import create_matcher_truncated

class PassivePyAnalyzer:
    
        """
            Get the data from a dataframe.

            Clean the dataset based on the given regex patterns.
            Match passive voice sentence level or corpus level.
            save the output to a file

        """
        def __init__(self, spacy_model = "en_core_web_lg"):

            """
            Create the Detector

            n_processses: number of core to use
            batch_size: size of batches of records passed onto the matcher
            regex_patterns: Patterns that should be detected and cleaned from the data
            
            
            """
            # print('installing the requirements...')
            # os.system('pip install -r https://raw.githubusercontent.com/mitramir55/PassivePy/main/PassivePyCode/PassivePySrc/requirements.txt')
            self.nlp, self.matcher_t = create_matcher_truncated(spacy_model)
            self.nlp, self.matcher_f = create_matcher_full(spacy_model)

        def parse_sentence(self, sentence):
            """
            This function allows us to see the components of a sentence, 
            specifically, the POS, DEP, and lemma
            """
            doc = self.nlp(sentence)
            all_matches = self.matcher(doc)
            
            
            for token in doc:
                print('word:', colored(token.text, 'green'), '\npos:', token.pos_,
                    '\ndependency:', token.dep_, '\ntag: ', token.tag_,
                    '\nlemma: ', token.lemma_)


            if all_matches:
                for id_, s,e in all_matches:
                    match_ = doc[s:e] 
                    print(match_)
                    print(colored(self.nlp.vocab.strings[id_], 'blue'))

        def detect_sents(self, cleaned_corpus, batch_size, n_process):

            print('Detecting Sentences...')

            """
            Separates sentences from each other in each record
             and puts them in a list along side the count of sentences in each 
             document in another list
             """
            cleaned_corpus = [corpus.lower() for corpus in cleaned_corpus]

            all_sentences = []
            count_sents = []
            unwanted = []
            puncs = set(string.punctuation)
            # start = time.process_time()

            m = 0
            for record_doc in tq.tqdm(self.nlp.pipe(cleaned_corpus, batch_size=batch_size, n_process = n_process), 
                                    leave=True,
                                    position=0,
                                    total=len(cleaned_corpus)):


                sentences = list(record_doc.sents)
                sentences = [str(sentence) if len(sentence)>=2 else 'Not a Sentence' for sentence in sentences] 


                for sentence in sentences:
                    i = sentences.index(sentence)
                
                    
                    #...........................joining with the previous one.............................#
                    # ones that start with but and their previous record doesn't have dot at its end
                    if i!=0:
                        if (re.search(r'^ *but', sentence) and not re.search(r'.$', sentences[i-1])) or all((re.search(r'^[A-Z0-9]', word) or re.search(r'^[\(\)\.\-]', word)) for word in sentence.split()) or re.search(r'^\(.*\)[\.\!\,]*', sentence):
                            j = 0
                            for j in range(1, i):
                                if i-j not in unwanted:
                                    sentences[i-j] = sentences[i-j] + sentences[i]
                                    unwanted.append(i)
                                    break
                            

                    #.........................joining with the next one..........................#
                    if i != len(sentences)-1:


                        if re.search(r', *$', sentence): # remove the one that's ended with comma
                            sentences[i] = ' '.join([sentences[i], sentences[i+1]])
                            unwanted.append(i+1)

                        if re.search(r'\- *$', sentence): 
                            # see if it's ended with hyphen then look at the next one
                            # if it has and in the beginning, forget about this one and go to the next to analyze the and 
                            # and not duplicate the process
                            if re.search(r'^ *(\([\w\. ]*\))* *and', sentences[i+1]):
                                continue
                            else: 
                                # but if there was no and in the next one,
                                #  join this with the next

                                sentences[i] = ' '.join([sentences[i], sentences[i+1]])
                                unwanted.append(i+1)
                        # see if it ends with and and join it with the 
                        elif re.search(r'and *$', sentence):
                            sentences[i] = ' '.join([sentences[i], sentences[i+1]])
                            unwanted.append(i+1)

                        # end with 'as well as' and join with the next one
                        elif re.search(r'((as well as) *)$', sentence):
                            sentences[i] = ' '.join([sentences[i], sentences[i+1]])
                            unwanted.append(i+1)

                        # end with the following phrasees and join with the next ones
                        elif re.search(r'((Exp\.)|(e\.g\.)|(i\.e\.))$', sentence):
                            sentences[i] = ' '.join([sentences[i], sentences[i+1]])
                            unwanted.append(i+1)


                m+=1
                for index in sorted(set(unwanted), reverse=True):
                    del sentences[index]
                unwanted = []

                
                count_sents.append(len(sentences))
                all_sentences.append(sentences) 

            all_sentences = list(chain.from_iterable(all_sentences))
            print(f'Total number of sentences = {len(all_sentences)}')


            # end = time.process_time()

            # calculating the time taken
            # taken_t = round(end-start, 2)
            # if taken_t < 60:
             #    print('time taken: ', taken_t, ' s')
            # else: print('time taken: ',  (taken_t)//60 , ' min ', round(taken_t%60, 1) , ' s\n')

            return np.array(count_sents, dtype='object'), np.array(all_sentences, dtype='object')


        def find_doc_idx(self, count_sents):

            """ finds the indices required for the documents and sentences"""

            m = 1
            sent_indices = []
            doc_indices = []
            for i in count_sents:
                n = 1
                for j in range(i):
                    sent_indices.append(n)
                    doc_indices.append(m)
                    n+=1
                m+=1
            return np.array(sent_indices), np.array(doc_indices)


        def add_other_cols(self, df, column_name, count_sents):

            """ creates a dataframe of all the other columns
            with the required number of repetitions for each """

            # create a list of all the col names
            fields = df.columns.tolist()
            # remove column_name
            del fields[fields.index(column_name)]

            other_columns = {}
            # create a df of all the other cols with 
            # appropriate number of repetitions
            for col in fields:
                properties = []
                for i in range(len(count_sents)):
                    properties.append(count_sents[i]*[df.loc[i, col]])
                
                properties = list(chain.from_iterable(properties))
                other_columns[col] = properties

            df_other_cols = pd.DataFrame.from_dict(other_columns)

            return df_other_cols  


        def match_text(self, cleaned_corpus, batch_size=1, n_process=1):

            """ This function finds passive matches in one sample sentence"""
            with HiddenPrints():
                # seperating sentences
                count_sents, all_sentences = self.detect_sents([cleaned_corpus], batch_size, n_process)

                matches, passive_c, binaries = self.find_matches(all_sentences, batch_size, n_process)
                

                s_output = pd.DataFrame(np.c_[all_sentences, binaries, matches, passive_c],
                            columns=['sentence', 'binary', 'passive_match(es)', 'raw_passive_count'])
                

                return s_output

        def find_matches(self, corpora, batch_size, n_process, truncated_passive=False):

            """ finds matches from each record """
            print(colored('Starting to find passives...', 'green'))  

            passive_c_f = []
            passive_c_t = []
            binaries_list_t = []
            binaries_list_f = []
            all_full_matches = []
            all_truncated_matches = []

            i = 0

            # stating with batches and n cores
            for doc in tq.tqdm(self.nlp.pipe(corpora, batch_size=batch_size, n_process=n_process), 
                                    leave=True,
                                    position=0,
                                    total=len(corpora)):

                binary_f = 0
                binary_t = 0
                
                # truncated passive voice ----------------------------------
                if truncated_passive==True:
                    match_t_i = []
                    all_matches_truncated = self.matcher_t(doc)

                    if all_matches_truncated:
                        spans = [doc[s:e] for id_, s,e in all_matches_truncated]

                        for span in spacy.util.filter_spans(spans):
                            match_t_i.append(str(span))

                        all_truncated_matches.append(match_t_i)
                        binary_t = 1
                        binaries_list_t.append(binary_t)

                # if there were no matches
                else:
                    all_truncated_matches.append(None)
                    passive_c_t.append(0)
                    binaries_list_t.append(binary_t)

                # full passive voice ----------------------------------------
                match_f_i = []
                all_matches_full = self.matcher_f(doc)

                # check for overlap
                if all_matches_full:
                    spans = [doc[s:e] for id_, s,e in all_matches_full]

                    for span in spacy.util.filter_spans(spans):
                        match_f_i.append(str(span))

                    binary_f = 1
                    all_full_matches.append(match_f_i)
                    passive_c_f.append(len(all_matches_full))
                    binaries_list_f.append(binary_f)

                # if there were no matches
                else:
                    all_full_matches.append(None)
                    passive_c_f.append(0)
                    binaries_list_f.append(binary_f)
                    

                i+=1


            if truncated_passive: 
                return np.array(all_full_matches, dtype='object'),\
                 np.array(all_matches_truncated, dtype='object'),\
                 np.array(passive_c_f, dtype='object'),\
                 np.array(passive_c_t, dtype='object'),\
                  np.array(binaries_list_f, dtype='object'),\
                  np.array(binaries_list_t, dtype='object')

            else :
                return np.array(all_full_matches, dtype='object'),\
                  np.array(passive_c_f, dtype='object'),\
                   np.array(binaries_list_f, dtype='object')
             

        def match_sentence_level(self, df, column_name, n_process = 1,
                                batch_size = 1000, add_other_columns=True,
                                truncated_passive=False):

            """
            finds matches based on sentences in all records and
            outputs a csv file with all the sentences in every document


            Parameters

            column_name: name of the column with text
            level: whether the user wants corpus level or sentence level
            results
            n_process: number of cores to use can be any number
            between 1 and the maximum number of cores available
            (set it to -1 to use all the cores available)
            batch_size: give records in batches to the matcher
            record when passed
            add_other_columns: True\False whether or not to add the other columns 
            to the outputted dataframe
            """
            
            df = df.reset_index(drop=True)
            # create a list of the column we will process
            cleaned_corpus = df.loc[:, column_name].values.tolist()

            # seperating sentences
            count_sents, all_sentences = self.detect_sents(cleaned_corpus, batch_size, n_process)

            # find indices required for the final dataset based on the document and sentence index
            sent_indices, doc_indices = self.find_doc_idx(count_sents)

            # create a df of matches -------------------------------------------
            if truncated_passive:
                all_full_matches, all_matches_truncated, passive_c_f, passive_c_t, binaries_list_f, binaries_list_t = self.find_matches(all_sentences, batch_size, n_process, truncated_passive)
                s_output = pd.DataFrame(np.c_[doc_indices, sent_indices, all_sentences, binaries_list_f, binaries_list_t, all_full_matches, all_matches_truncated, passive_c_f, passive_c_t],
                            columns=['docId', 'sentenceId', 'sentence', 'binary_full_passive', 'binary_truncated_passive', 'full_passive_match(es)', 'truncated_passive_match(es)',
                             'raw_full_passive_count', 'raw_truncated_passive_count'])
            else:
                all_full_matches, passive_c_f, binaries_list_f = self.find_matches(all_sentences, batch_size, n_process, truncated_passive)
                s_output = pd.DataFrame(np.c_[doc_indices, sent_indices, all_sentences, binaries_list_f, all_full_matches, passive_c_f],
                            columns=['docId', 'sentenceId', 'sentence', 'binary_full_passive', 'full_passive_match(es)', 'raw_full_passive_count'])


            # concatenating the results with the initial df -------------------
            if add_other_columns==True:

                other_cols_df = self.add_other_cols(df, column_name, count_sents)
                assert len(other_cols_df) == len(s_output)
                df_final = pd.concat([s_output, other_cols_df], axis = 1)

                return df_final

            else:
                return s_output



        def match_corpus_level(self, df, column_name, n_process = 1,
            batch_size = 1000, add_other_columns=True,
            percentage_of_passive_sentences = True, truncated_passive=False):

            """
            finds matches based on sentences in all records and
            outputs a csv file with all the sentences in every document


            Parameters

            column_name: name of the column with text
            level: whether the user wants corpus level or sentence level
            results
            n_process: number of cores to use can be any number
            between 1 and the maximum number of cores available
            (set it to -1 to use all the cores available)
            batch_size: give records in batches to the matcher
            record when passed
            add_other_columns: True\False whether or not to add the other columns 
            to the outputted dataframe
            sentences to the output dataset
            """
            
            df = df.reset_index(drop=True)
            # create a list of the column we will process
            cleaned_corpus = df.loc[:, column_name].values.tolist()


            if percentage_of_passive_sentences:
                s_output = self.match_sentence_level(df, column_name, n_process = n_process,
                                batch_size = batch_size, add_other_columns=add_other_columns)

                full_passive_matches = []
                truncated_passive_matches = []
                full_passives_c = []
                truncated_passives_c = []
                full_binaries = []
                truncated_binaries = []
                full_passive_percentages = []
                truncated_passive_percentages = []
                count_sents = []
                count_f_p_sents = []
                count_t_p_sents = []
                ids_ = s_output.docId.unique()


                for i in tq.tqdm(ids_, leave=True, position=0, total=len(ids_)):

                    # select all the sentences of a doc
                    rows = s_output[s_output['docId'] == i]

                    # concatenate all the proberties ------------------------------------
                    count_sents.append(len(rows))

                    # full passive
                    count_full_passive_s = sum(rows.binary_full_passive)
                    count_f_p_sents.append(count_full_passive_s)
                    percent_full =  count_full_passive_s/ len(rows)
                    full_passive_percentages.append(percent_full)

                    # truncated passive
                    if truncated_passive:
                        count_truncated_passive_s = sum(rows.binary_truncated_passive)
                        count_t_p_sents.append(count_truncated_passive_s)
                        percent_truncated =  count_truncated_passive_s/ len(rows)
                        truncated_passive_percentages.append(percent_truncated)
                        # binary will be =1 if there is even one 1 
                        if any(rows.binary_full_passive) == 1:
                            truncated_binaries.append(1)
                        else: truncated_binaries.append(0)



                    # binary will be =1 if there is even one 1 
                    if any(rows.binary_full_passive) == 1:
                        full_binaries.append(1)
                    else: full_binaries.append(0)

                    # putting all sentences' passives in one list ----------------------------
                    # full passive
                    full_passives = [val for val in rows['full_passive_match(es)'].values if val!=None]
                    full_passives = list(chain.from_iterable(full_passives))
                    full_passive_matches.append(full_passives)
                    full_passives_c.append(len(full_passives))

                    # truncated passive
                    if truncated_passive:
                        truncated_passives = [val for val in rows['truncated_passive_match(es)'].values if val!=None]
                        truncated_passives = list(chain.from_iterable(truncated_passives))
                        truncated_passive_matches.append(truncated_passives)
                        truncated_passives_c.append(len(truncated_passives))
                # Full passive
                full_passives_c = np.array(full_passives_c, dtype='object')
                full_passive_matches = np.array(full_passive_matches, dtype='object')
                full_passive_percentages = np.array(full_passive_percentages, dtype='object')
                full_binaries = np.array(full_binaries, dtype='object')

                # truncated passive
                truncated_passives_c = np.array(truncated_passives_c, dtype='object')
                truncated_passive_matches = np.array(truncated_passive_matches, dtype='object')
                truncated_passive_percentages = np.array(truncated_passive_percentages, dtype='object')
                truncated_binaries = np.array(truncated_binaries, dtype='object')


                count_sents = np.array(count_sents, dtype='object')
                cleaned_corpus = np.array(cleaned_corpus, dtype='object')

                if truncated_passive:
                    truncated_passives_c = np.array(truncated_passives_c, dtype='object')
                    truncated_passives = np.array(truncated_passives, dtype='object')

                
                assert len(cleaned_corpus) == len(binaries) == len(percentages) == len(matches) == len(full_passives_c) == len(count_p_sents)
                if truncated_passive:
                    d_output = pd.DataFrame(np.c_[cleaned_corpus, binaries, matches, full_passives_c, count_p_sents, count_sents, percentages],
                                            columns=['document', 'binary', 'full_passive_match(es)', 'raw_full_passive_count', 'raw_passive_sents_count', 'raw_sentence_count', 'passive_sents_percentage' ])
                else:
                    d_output = pd.DataFrame(np.c_[cleaned_corpus, binaries, matches, full_passives_c, count_p_sents, count_sents, percentages],
                                        columns=['document', 'binary', 'full_passive_match(es)', 'raw_full_passive_count', 'raw_passive_sents_count', 'raw_sentence_count', 'passive_sents_percentage' ])



            elif percentage_of_passive_sentences==False:
                matches, passive_c, binaries = self.find_matches(cleaned_corpus, batch_size, n_process)
                d_output = pd.DataFrame(np.c_[cleaned_corpus, binaries, matches, passive_c],
                                        columns=['Document', 'binary', 'full_passive_match(es)', 'raw_passive_count' ])


            assert len(cleaned_corpus) == len(matches) == len(passive_c)

            
            # now we have all the matches we just have to
            # create a dataframe for the results
            if add_other_columns==True:

                # create a list of all the col names
                fields = df.columns.tolist()
                # remove column_name
                del fields[fields.index(column_name)]

                

                assert len(df[fields]) == len(d_output)

                d_output = pd.concat([d_output, df[fields]], axis = 1)
                

            


            
            return d_output


# for stopping the print statements in one sample sentences
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
