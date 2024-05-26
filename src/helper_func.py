from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
from sklearn.model_selection import cross_val_score
import gensim
import re
import os, sys
## custom packages
src_dir = os.path.join( 'src')
sys.path.append(src_dir)

from filter_words import run_stopword_statistics
from filter_words import make_stopwords_filter
from filter_words import remove_stopwords_from_list_texts

import spacy
nlp = spacy.load('en_core_web_md')
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import seaborn as sns

def clean_stopwords(texts, stop_words, processing_choice='nouns', N_s = 100, 
                    path_stopword_list =  os.path.join('data','stopword_list_en'),
                    method = 'INFOR', cutoff_type = 'p',
                    cutoff_val = 0.1):
    """
    args:
    processing_choice: 'nouns', 'with adj', 'noun chunks', 'all'
    method: 'INFOR', 'BOTTOM', 'TOP', 'TFIDF', 'TFIDF_r', 'MANUAL'
    """
    processed_text = process_words(texts = texts, stop_words = stop_words, paradigm = processing_choice)
    
    df_sw = run_stopword_statistics(processed_text,N_s=N_s,path_stopword_list=path_stopword_list)
    
    df_filter = make_stopwords_filter(df_sw,
                                  method = method,
                                  cutoff_type = cutoff_type, 
                                  cutoff_val = cutoff_val, )
    
    list_words_filter = list(df_filter.index)
    
    list_texts_filter = remove_stopwords_from_list_texts(processed_text, list_words_filter)
    
    N = sum([ len(doc) for doc in processed_text ])
    N_filter = sum([ len(doc) for doc in list_texts_filter ])
    print('Remaining fraction of tokens: ',N_filter/N)
    
    texts_nouns = [" ".join(i) for i in list_texts_filter]
    
    dictionary_filter = gensim.corpora.Dictionary(list_texts_filter)
    
    corpus_filter = [dictionary_filter.doc2bow(t) for t in list_texts_filter]
    
    n_features = len(dictionary_filter)
    
    return {'texts_filter': texts_nouns,
            'list_texts_filter': list_texts_filter,
           'dictionary_filter': dictionary_filter,
           'corpus_filter': corpus_filter,
           'n_features': n_features}

def coherence_per_topic(estimator, X,y=None):
        
    topics = get_topics_from_model(
        estimator,
        vectorizer,
        n_top_words
    )
    cm = CoherenceModel(
        topics=topics,
        texts = output['list_texts_filter'],
        corpus=output['corpus_filter'], 
        dictionary=output['dictionary_filter'], 
        coherence='c_v', 
        topn=n_top_words
    )
    
    return cm.get_coherence_per_topic(segmented_topics=None, with_std=True, with_support=False)

def find_best_n_topics(model_name, texts, dictionary, corpus, n_features, n_top_words, coherence):
    # Create a list of the topic numbers we want to try
    topic_nums = list(np.arange(10, 500 + 1, 40))
    
    texts_nouns = [" ".join(i) for i in texts]

    # Run the nmf model and calculate the coherence score
    # for each number of topics
    coherence_scores = []

    for num in topic_nums:
        
        if model_name == 'nmf':
            # For NMF
            vectorizer = TfidfVectorizer(
                #max_features=n_features, stop_words=list(stopwords)
            )

            tfidf = vectorizer.fit_transform(texts_nouns)

            model = NMF(n_components=num,
                        max_iter=200, 
                        init="nndsvda", 
                        random_state=0, 
                        alpha_W=1, 
                        l1_ratio=0.2).fit(tfidf)
            
        elif model_name == 'kullback-leibler':
            # For NMF
            vectorizer = TfidfVectorizer(
                #max_features=n_features, stop_words=list(stopwords)
            )

            tfidf = vectorizer.fit_transform(texts_nouns)

            model = NMF(
                n_components=num,
                random_state=25,
                beta_loss="kullback-leibler",
                init="nndsvda",
                solver="mu",
                max_iter=200,
                alpha_W=1,
                l1_ratio=0.2,
            ).fit(tfidf)
        
        elif model_name == 'lda':
            ## for LDA

            vectorizer = CountVectorizer(
                max_features=n_features, stop_words=list(stopwords),
            )

            tf = vectorizer.fit_transform(texts_nouns)

            model = LatentDirichletAllocation(
                n_components=num,
                max_iter=10,
                learning_method="online",
                learning_offset=150,
                random_state=0,
            ).fit(tf)
        
        elif model_name =='gensim_lda':
            model = gensim.models.ldamodel.LdaModel(corpus=corpus, 
                id2word=dictionary,
                num_topics=num, 
                random_state=117, 
                #update_every=1,
                #chunksize=1500, 
                #passes=5, iterations=100,
                #alpha='asymmetric', eta=1/100,
                #alpha='auto', eta=1/100,
                #per_word_topics=True
               )
            
        elif model_name =='gensim_nmf':
            model = gensim.models.Nmf(corpus=corpus, 
                num_topics=num,
                random_state =1
               )         
        



        # Run the coherence model to get the score
        if model_name =='gensim_lda' or model_name =='gensim_nmf':            
            
            cm = CoherenceModel(
                model = model,
                texts = texts,
                corpus=corpus, 
                dictionary=dictionary, 
                coherence=coherence,
                topn = n_top_words
            )
            
        else:        
        
            topics = get_topics_from_model(
                model,
                vectorizer,
                n_top_words
            )
            cm = CoherenceModel(
                topics=topics,
                texts = texts,
                corpus=corpus, 
                dictionary=dictionary, 
                coherence=coherence,
                topn=n_top_words
            )
        
        
        coherence_scores.append(round(cm.get_coherence(), 5))

    # Get the number of topics with the highest coherence score
    return list(zip(topic_nums, coherence_scores))


    

def get_clean_output(list_texts_filter):
    
    texts_nouns = [" ".join(i) for i in list_texts_filter]
    
    dictionary_filter = gensim.corpora.Dictionary(list_texts_filter)
    
    corpus_filter = [dictionary_filter.doc2bow(t) for t in list_texts_filter]
    
    n_features = len(dictionary_filter)
    
    return {'texts_filter': texts_nouns,
            'list_texts_filter': list_texts_filter,
           'dictionary_filter': dictionary_filter,
           'corpus_filter': corpus_filter,
           'n_features': n_features}

def get_list(texts, stop_words, processing_choice='nouns', N_s = 100,
                    path_stopword_list =  os.path.join('data','stopword_list_en'),
                    method = 'INFOR', cutoff_type = 'p',
                    cutoff_val = 0.1, path_to_file = 'data/list_filtered_'):
    """
    args:
    processing_choice: 'nouns', 'with adj', 'noun chunks', 'all'
    method: 'INFOR', 'BOTTOM', 'TOP', 'TFIDF', 'TFIDF_r', 'MANUAL'
    """
    
    if os.path.exists('data/list_filtered_' + processing_choice + '_' +str(cutoff_val) + '.csv') and path_to_file == 'data/list_filtered_':
        lst = pd.read_csv('data/list_filtered_' + processing_choice + '_' +str(cutoff_val) + '.csv').drop('Unnamed: 0', axis=1).fillna('').values.tolist()
        
        return [list(filter(None, i)) for i in lst]
    
    else:
        
        processed_text = process_words(texts = texts, stop_words = stop_words, paradigm = processing_choice)

        df_sw = run_stopword_statistics(processed_text,N_s=N_s,path_stopword_list=path_stopword_list)

        df_filter = make_stopwords_filter(df_sw,
                                      method = method,
                                      cutoff_type = cutoff_type, 
                                      cutoff_val = cutoff_val, )

        list_words_filter = list(df_filter.index)

        list_texts_filter = remove_stopwords_from_list_texts(processed_text, list_words_filter)


        N = sum([ len(doc) for doc in processed_text ])
        N_filter = sum([ len(doc) for doc in list_texts_filter ])
        print('Remaining fraction of tokens: ',N_filter/N)

        pd.DataFrame(list_texts_filter).to_csv(path_to_file + processing_choice + '_' +str(cutoff_val) + '.csv')

        return list_texts_filter

def get_top_n_words(corpus, n=None, vect = 'CountVectorizer'):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    
    get_top_n_words(["I love Python", "Python is a language programming", "Hello world", "I love the world"], 3) -> 
    ['love', 'python', 'world']
    """
    
    if vect == 'CountVectorizer':
        vec = CountVectorizer().fit(corpus)
    else: 
        vec = TfidfVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return [i[0] for i in words_freq[:n]]

def get_topics_from_model(model, vect, n_top_words):
    feature_names = vect.get_feature_names() # get_feature_names() deprecated    
    #feature_names = vect.get_feature_names_out()
    num = len(model.components_)
    
    topics = []

    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        topics.append(top_features)
        
    return topics

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def my_lemmatizer(doc):
    ## using spacy lemmatizer
    doc_cleaned = ' '.join(re.findall(r'\b\w[\w\']+\b', doc))
    return [ w.lemma_.lower() for w in nlp(doc_cleaned) 
                      if w.lemma_ not in ['_', '.', '-PRON-'] ]

def plot_top_words(model, feature_names, n_top_words, title):
    num = len(model.components_)
    h = math.ceil(num/5)
    f = 8*h
    fig, axes = plt.subplots(h, 5, figsize=(30, f), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        #ax.set_title(f"Topic {topic_idx}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        #fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
    
def plot_top_words_colors(model, feature_names, n_top_words, title):
    num = len(model.components_)
    h = math.ceil(num/5)
    f = 8*h
    fig, axes = plt.subplots(h, 5, figsize=(30, f), sharex=True)
    axes = axes.flatten()
    
    palette = sns.color_palette("Spectral", 10).as_hex()
    
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7, color = palette[topic_idx])
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        #fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

def process_words(texts, stop_words, paradigm = 'nouns'):
    result = []
    for t in texts:
        t = ' '.join([i for i in re.findall(r'\b\w[\w\']+\b', t) if (not has_numbers(i)) & (len(i)>2)])
        doc = nlp(t)
        if paradigm == 'nouns':
            # keep only nous
            result.append([token.lemma_.lower() for token in doc 
                           if (token.pos_ in ['NOUN', 'PROPN']) & (token.lemma_.lower() not in stop_words) and not token.text.isupper() and not token.is_stop])
        elif paradigm == 'with adj':
            # keep nous and adjectives
            result.append([token.lemma_.lower() for token in doc 
                           if (token.pos_ in ['NOUN', 'PROPN', 'ADJ']) & (token.lemma_.lower() not in stop_words and not token.text.isupper() and not token.is_stop)])
        elif paradigm == 'noun chunks': 
            # keep noun chunks
            result.append([token.lemma_.lower() for token in doc.noun_chunks if (token.lemma_.lower() not in stop_words) and not token.text.isupper() and not token.is_stop])
        else:
            # keep all
            result.append([token.lemma_.lower() for token in doc if (token.lemma_.lower() not in stop_words) and not token.text.isupper() and not token.is_stop])
            
    if paradigm == 'noun chunks':
        return [" ".join(i).split(' ') for i in result] 
    else:
        return result
    
