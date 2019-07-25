# 과거 3.2.5 버전을 통해 moses 분절 사용
!pip install nltk==3.2.5

# 필요 패키지 불러오기

import sys, fileinput
from nltk.tokenize.moses import MosesTokenizer

# MosesTokenizer 객체화
tokenizer = MosesTokenizer

if __name__ == '__main__':
    for line in fileinput.input():
        ## 공백이 아닐 경우
        if line.strip() != '':
            tokens = tokenizer.tokenize(line.strip(), escape=False)
            ## 토큰 사이 구분은 ' ' 공백으로
            sys.stdout.write(' '.join(tokens) + '\n')
        else:
            sys.stdout.write('\n')
