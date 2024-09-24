import spacy

import numpy as np
import networkx as nx




nlp = spacy.load("en_core_web_sm")

text = '''Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.[32]\n\nPython is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured (particularly procedural), object-oriented and functional programming. It is often described as a "batteries included" language due to its comprehensive standard library.[33][34]\n\nGuido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language and first released it in 1991 as Python 0.9.0.[35] Python 2.0 was released in 2000. Python 3.0, released in 2008, was a major revision not completely backward-compatible with earlier versions. Python 2.7.18, released in 2020, was the last release of Python 2.[36]\n\nPython consistently ranks as one of the most popular programming languages, and has gained widespread use in the machine learning community.'''




def sentence_similarity(sent1, sent2):
    doc1 = nlp(sent1)
    doc2 = nlp(sent2)

    return doc1.similarity(doc2) # cosine similarity



def calculate_similarity_matrix(sentences):
    num_sentences = len(sentences) # 8
    similarity_matrix = np.zeros((num_sentences, num_sentences)) # I create an empty numpy 2d array of 8 * 8 = 64 filled with zeros
    for i, sent1 in enumerate(sentences): # enumerating the sentence list --> returns index,sentence for rows
        for j, sent2 in enumerate(sentences): # for columns
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sent1, sent2)
    return similarity_matrix


def summarize_text_pagerank(text, ratio=0.6):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    if not sentences:
        raise ValueError("No sentences found in the input text.")
    # Create similarity matrix
    similarity_matrix = calculate_similarity_matrix(sentences)

    # Build graph and calculate PageRank
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)


    # Sort sentences by score and select top sentences
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    num_sentences = int(len(sentences) * ratio)
    summary_sentences = [s for _, s in ranked_sentences[:num_sentences]]
    summary = " ".join(summary_sentences)
    doc_len = len(text)
    summ_len = len(summary)
    return summary,summ_len,doc_len


