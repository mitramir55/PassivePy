def corous_level_additional_info(df, colName):


    df = df.reset_index(drop=True)
    # create a list of the column we will process
    cleaned_corpus = df.loc[:, colName].values.tolist()

    # seperating sentences
    count_sents, all_sentences = self.detect_sents(cleaned_corpus, batch_size, n_process)

    # find indices required for the final dataset
    # based on the document and sentence index
    sent_indices, doc_indices = self.find_doc_idx(count_sents)
