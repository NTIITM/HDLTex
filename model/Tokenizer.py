"""
这里做一个分词器
为了使其方便可用
输入为文本，输出为words
基于nltk
"""

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

class tokenizer(object):
    def __init__(self,
                 stemer=PorterStemmer(),
                 interpunctuations=None, ):
        if interpunctuations is None:
            interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        self.stemer = stemer
        self.interpunctuations = interpunctuations
        self.word_to_id = {}
        self.id_to_word = {}

    def fit_on_text(self,
                 s: str) -> list[str]:
        # 以空格形式实现分词,除去停用词等可以动态加入
        tokens = word_tokenize(s)
        tokens = [token for token in tokens if
                  token not in self.interpunctuations and token not in stopwords.words("english")]
        stem_word = [self.stemer.stem(token) for token in tokens]
        for word in stem_word:
            if word not in self.word_to_id:
                new_id = len(self.word_to_id)
                self.word_to_id[word] = new_id
                self.id_to_word[id] = word
        # sentence = sent_tokenize(s)
        return stem_word
    def convert_id


if __name__ == '__main__':
    paragraph = "The first time I heard that song was in Hawaii on radio. I was just a kid, and loved it very much! What a fantastic song!"
    tz = Tokenizer(num_words=50)
    tz.fit_on_texts(paragraph)

    print(tz.texts_to_sequences(paragraph))
