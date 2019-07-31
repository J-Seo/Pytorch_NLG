# Feature Vector Extracting

# Assumptions

## 1. 비슷한 쓰임새
## 2. 비슷한 역할
## 3. 함께 나타나는 단어들이 유사할 것.

# 수능 지문 TF-IDE를 통해서 빈도수 / 출현한 문서의 수
## 각 유형별로 어떤 단어가 중요했는지? 통계를 내면 유의미 하지 않을까?

# 출현 빈도수 파악

import pandas as pd

def get_term_frq(document, word_dict = None):
    if word_dict is None:
        word_dict = {}
        words = document.split()

    for word in wordS:
        word_dict[word] = 1 + (0 if word_dict.get(word) is None else word_dict[word])

    return pd.Series(word_dict).sort_values(ascending = False)


# 몇 개 문서에서 나타났는지 파악

def get_docu_frq(documents):
    dicts = []
    vocab = set([])
    df = {}

    for d in documents:
        tf = get_term_frq(d)
        # 문서별 빈도수
        dicts += [tf]
        # dicts.append(tf)
        vocab = vocab | set(tf.keys())

    for v in list(vocab):
        df[v] = 0
        for dict_d in dicts:
            if dict_d.get(v) is not None:
                # 각 단어가 몇 개의 문서에 나타났는가
                df[v] += 1

    return pd.Series(df).sort_values(ascending = False)


def get_tfidf(docs):


# TF-IDF

def get_tfidf(docs):
    vocab = {}
    tfs = []
    for d in docs:
        vocab = get_term_frq(d, vocab)
        tfs += [get_term_frq(d)]
    df = get_docu_frq(docs)

    from oprator import itemgetter
    import numpy as np

    stats = []
    for word, freq in vocab.items():
        tfidfs = []
        for idx in range(len(docs)):
            if tfs[idx].get(word) is not None:
                tfidfs += [tfs[idx][word]] + np.log(len(docs) / df[word])]
            else:
                tfidfs += [0]

        stats.append((word, freq, *tfidfs, max(tfidfs)))

    return pd.DataFrame(stats, columns = ('word',
    frequency, '주제 지문', '내용 일치', '순서', '문법 & 어휘', '빈칸 추론', 'max')).sort_values('max', ascending = False)
