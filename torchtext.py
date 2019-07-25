!pip install torchtext

# 코퍼스와 레이블 읽기
## 탭 '\t'로 구분된 데이터의 입력을 받는 경우

from torchtext import unicodedata

class DataLoader(object):

    def __init__(self, train_fn, valid_fn,
                batch_size = 64,
                device = -1,
                max_vocab = 99999,
                min_freq = 1,
                use_eos = False,
                shuffle = True):
        super(DataLoader, self).__init__()

        # 입력 파일에 대한 필드를 정의
        # 2개의 필드로 구성

        self.label = data.Field(sequential = False,
                                use_vocab = True,
                                unk_token = None)
        self.text = data.Field(use_vocab = True,
                                batch_first = True,
                                include_lengths = False,
                                eos_token = '<EOS>' if use_eos else None)

        # 두 개의 열 (필드)는 tab으로 구분
        # TabularDataset을 이용해서 두 개의 열은 입력 파일로 로드
        # train_fn, vaild_fn으로 구분
        # 파일은 label field와 text field로 구분

        train, valid = data.TabularDataset.splits(path = '',
                                                train = train_fn,
                                                validation = valid_fn,
                                                format = 'tsv',
                                                fields = [('label', self.label),
                                                ('text', self.text)])

        # 로드된 데이터셋은 각각 이터레이터 연산
        # train iterator & valid iterator
        # 길이에 의거하여 입력 문장 분류, 유사한 문장을 그룹핑

        self.train_iter, self.valid_iter = data.BucketIterator.splits((train, valid),
        batch_size = batch_size, shuffle = shuffle, sort_key = lambda x: len(x.text), sort_within_batch = True)

        # label과 text field에 맞는 vocab 구성
        # mapping table btw words and indices
        self.label.build_vocab(train)
        self.text.build_vocab(train, max_size = max_vocab, min_freq = min_freq)
