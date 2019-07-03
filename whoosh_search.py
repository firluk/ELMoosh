import argparse
import csv
import json
import os.path
import random

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from nltk import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, syntax

# Download to get punkt package
# nltk.download()

# Constants
LIMIT = 1
REGEX_CAMEL_CASE_MATCH = "^[a-zA-Z]+([A-Z][a-z]+)+$"
PULSE_RATE = 0.1

# Globals
__indexdir__ = "indexdir"
__wikipassage_data__ = None  # PH for the whole wikipassage data as navigable text reference
__question_answer_data__ = None

# tf-hub stuff
# elmo = hub.Module("https://tfhub.dev/google/elmo/2")
elmo = hub.Module("module/module_elmo2/", trainable=False)

# tf session config - GPU: 0 to enforce tf to use cpu
config = tf.ConfigProto(
    device_count={'GPU': 0}
)


# Project functions
def get_wikipassage_data_object():
    global __wikipassage_data__
    if __wikipassage_data__ is None:
        with open('./WikiPassageQA/document_passages.json') as json_file:
            __wikipassage_data__ = json.load(json_file)
    return __wikipassage_data__


def get_question_answer_data_object():
    global __question_answer_data__
    if __question_answer_data__ is None:
        __question_answer_data__ = []
        tsv_file_name = "test-sample.tsv"
        with open(os.path.join("./WikiPassageQA/", tsv_file_name)) as tsv_file:
            reader = csv.reader(tsv_file, delimiter='\t')
            for row in reader:
                __question_answer_data__.append(row)
    return __question_answer_data__


def create_schema():
    # we want to utilize stemmed text for retrieval, and analyze further un-stemmed with ELMo
    return Schema(document_id=ID(stored=True),  # document id
                  body=TEXT(analyzer=StemmingAnalyzer()))  # stemmed body of the document


def create_index(schema=create_schema(), local_indexdir=__indexdir__):
    # create standard index dir in the working directory
    if not os.path.exists(local_indexdir):
        os.mkdir(local_indexdir)

    # create index under indexdir
    ix = index.create_in(local_indexdir, schema)

    return ix


def load_index(local_indexdir=__indexdir__):
    # loads/opens index directory
    if args.index:
        create_searchable_data()
    return index.open_dir(local_indexdir)


def document_concat(document):
    # concatenates a dictionary to a single string
    concat = []
    for passage in (v for k, v in sorted(document.items(), key=lambda e: e[0])):
        concat.append(passage)
    return " ".join(concat)


def create_searchable_data():
    create_index()
    ix = index.open_dir(__indexdir__)
    writer = ix.writer()
    wikipassage_dict = get_wikipassage_data_object()
    pulse = 1
    total = len(wikipassage_dict.items())
    print("Concatenating passages to a single document")
    for document_id, document_dict in wikipassage_dict.items():
        if pulse == 1 or pulse == total or (pulse % int(total * PULSE_RATE) == 0):
            print(str(pulse) + " / " + str(total))
        body = document_concat(document_dict)
        writer.add_document(document_id=document_id,
                            body=body)
        pulse += 1
    print("Finishing writing and saving all additions and changes to disk...")
    writer.commit()
    print("Done")


def search_for_documents(query_str):
    ix = load_index()
    wikipassage_obj = get_wikipassage_data_object()
    relevant_documents = []
    with ix.searcher() as searcher:
        query_parser = QueryParser("body", schema=ix.schema, group=syntax.OrGroup)
        query = query_parser.parse(query_str)
        results = searcher.search(query, limit=LIMIT)
        for result in results:
            document_id = result['document_id']
            score = result.score
            passages = wikipassage_obj[document_id]
            relevant_documents.append(document_id)
            print(document_id)
            print(str(score))
            print(passages)
    return relevant_documents


def search_for_passages(doc_id, question, embedded=False):
    if not embedded:
        embedded_question = embed_question(question)
    else:
        embedded_question = question

    document = get_document(str(doc_id))
    passages_with_ids = document.items()
    similarities = []
    for id, passage in passages_with_ids:
        embedded_passage = embed_passage(passage)
        similarity_matrix = cosine_similarity(embedded_question, embedded_passage, dense_output=False)
        similarities.append((id, np.mean(similarity_matrix)))

    return sorted(similarities, key=lambda e: e[1])[0:5]


def embed_passage(passage):
    """
    Embeds passage to elmo representation
    :param passage:
    :return:
    """
    sentences = sent_tokenize(passage)
    embedded_sentences = elmo(sentences,
                              signature="default",
                              as_dict=True)["elmo"]
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        embedded_sentences = sess.run(embedded_sentences)

    embedded_sentences = np.sum(embedded_sentences, axis=(1))
    return embedded_sentences


def embed_question(question):
    """
    Embeds single sentence to elmo representation
    :param question:
    :return:
    """
    embedded_question = elmo(question,
                             signature="default",
                             as_dict=True)["elmo"]
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        embedded_question = sess.run(embedded_question)

    embedded_question = np.sum(embedded_question, axis=(1))
    return embedded_question


def get_document(doc_id):
    obj = get_wikipassage_data_object()
    return obj[doc_id]


def get_passage(doc_id, passage_id):
    return get_document(doc_id)[passage_id]


def random_passage():
    """
    Aux function for fetching a random passage
    :return:
    """
    obj = get_wikipassage_data_object()
    doc_id = str(random.randint(1, len(obj.values())))
    passage_id = str(random.randint(1, len(obj[doc_id].values())))
    passage = obj[doc_id][passage_id]
    return passage


def get_question(row_number):
    """
    Aux function for fetching a random question
    :return: row ['QID', 'Question', 'DocumentID', 'DocumentName', 'RelevantPassages']
    """
    obj = get_question_answer_data_object()
    return obj[row_number]


def random_question():
    """
    Aux function for fetching a random question
    :return: row ['QID', 'Question', 'DocumentID', 'DocumentName', 'RelevantPassages']
    """
    obj = get_question_answer_data_object()
    row_number = random.randint(1, len(obj) - 1)
    return get_question(row_number)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", help="Create index anew", action="store_true")
    args = parser.parse_args()

    question_row = random_question()
    question_id = question_row[0]
    question = question_row[1]

    correct_answer = (question_row[2], question_row[4])
    print(question)
    print(correct_answer)
    print(question_row)

    document_ids = search_for_documents(question)

    embedded_question = embed_question([question])
    document_similarities = []
    answers = {"id": question_id, "answers": []}

    for document_id in document_ids:
        passage_similarities = search_for_passages(document_id, embedded_question, True)
        document_similarities.append(passage_similarities)
        document_similarities.sort(key=lambda doc_sim: doc_sim[0][1])
        answer = {}

    print("Script end")
