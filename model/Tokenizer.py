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
        self.word_to_id = {
            '<unk>': 0
        }
        self.id_to_word = {
            0: '<unk>'
        }

    # todo 后续考虑加入仅有出现次数足够多的word放进去
    def fit_on_text(self,
                    s: str) -> None:
        # 以空格形式实现分词,除去停用词等可以动态加入
        s = s.lower()
        tokens = word_tokenize(s)
        tokens = [token for token in tokens if
                  token not in self.interpunctuations and token not in stopwords.words("english")]
        stem_word = [self.stemer.stem(token) for token in tokens]
        for word in stem_word:
            if word not in self.word_to_id:
                new_id = len(self.word_to_id)
                self.word_to_id[word] = new_id
                self.id_to_word[new_id] = word
        # sentence = sent_tokenize(s)
        return

    def fit_on_texts(self, texts, num_accept=1000):
        from collections import Counter
        tem = []
        for text in texts:
            tokens = word_tokenize(text)
            tem.extend(tokens)
        d = Counter(tem)
        # print(d)
        s_d = list(sorted(d.items(), key=lambda x: -x[1]))
        for word, _ in s_d:
            new_id = len(self.word_to_id)
            if new_id == num_accept:
                break
            self.word_to_id[word] = new_id
            self.id_to_word[new_id] = word

    def convert_word_to_id(self, words: list[str]) -> list[int]:
        ret = []
        for word in words:
            word = word.lower()
            tem_word = self.stemer.stem(word)
            if tem_word in self.word_to_id:
                # ipdb.set_trace()
                ret.append(self.word_to_id[tem_word])
            # else:
            #     ret.append(self.word_to_id['<unk>'])
        return ret

    def convert_id_to_word(self, ids: list[int]) -> list[str]:
        ret = []
        for id in ids:
            assert id in self.id_to_word
            ret.append(self.id_to_word[id])
        return ret


if __name__ == '__main__':
    paragraph = "The first time I heard that song was in Hawaii on radio. I was just a kid, and loved it very much! What a fantastic song!"
    print(paragraph)
    tz = tokenizer()
    tz.fit_on_texts([paragraph])
    print(tz.word_to_id)
    # ids = tz.convert_word_to_id(word_tokenize(paragraph))
    # print(ids)
    # words = tz.convert_id_to_word(ids)
    # print(" ".join(words))
