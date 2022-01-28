'''01) 토큰화 Tokenization

1. 단어 토큰화 : 보통은 띄어쓰기 단위지만 구두점이나 특수문자도 고려해야 함

2. 토큰화 중 생기는 선택의 순간 : don't'''

from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence

print('단어 토큰화1 :',word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
# word_tokenize : Don't -> Do, n't / Jone's -> Jone, 's

print('단어 토큰화2 :',WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
# wordPuncTokenizer : Don't -> Don, ', t

print('단어 토큰화3 :',text_to_word_sequence("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
# keras text_to_word_sequence : don't -> don't

'''3. 토큰화 고려 사항

- 구둣점이나 특수문자 단순 제외 불가
- 줄임말과 단어 내 띄어쓰기 

4. 문장 토큰화'''

# 한국어 KSS(Korean Sentence Splitter)

import kss # pip install kss # 아나콘다에 없음

text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'
print('한국어 문장 토큰화 :',kss.split_sentences(text))

'''5. 한국어에서 토큰화의 어려움 : 어절

- 교착어 : 형태소
- 띄어쓰기의 자유

6. 품사태깅

7. 한국어 토큰화 실습'''

from konlpy.tag import Okt
from konlpy.tag import Kkma

okt = Okt()
kkma = Kkma()

print('OKT 형태소 분석 :',okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 품사 태깅 :',okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 명사 추출 :',okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요")) 

print('꼬꼬마 형태소 분석 :',kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('꼬꼬마 품사 태깅 :',kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('꼬꼬마 명사 추출 :',kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  


'''02) 정제 Cleaning 과 정규화 Normalization

정제 : 갖고 있는 코퍼스의 노이즈 제거
정규화 : 표현방법이 다른 단어들을 통합시켜 같은 단어로 만듦

1. 규칙에 기반한 표기가 다른 단어들 통합

USA, US -> USA

2. 대, 소문자 통합

3. 불필요한 단어 제거 : 빈도 적거나 길이 짧음 (ex. 불용어)

4. 정규표현식 : 전처리에 이용'''


'''03) 어간 추출 Stemming 과 표제어 추출 Lemmatization

1. 표제어 추출 (기본 사전형 단어 is, are, am -> be)

어간 : 의미 담은 부분
접사 : 단어에 추가적인 의미줌'''

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']

print('표제어 추출 전 :',words)
print('표제어 추출 후 :',[lemmatizer.lemmatize(word) for word in words])

'''2. 어간 추출

어간 추출 : am -> am / having -> hav

표제어 추출 : am -> be / having -> have

3. 한국어 어간 추출

어간 : 원칙적으로 모양이 바뀌지 않는 부분 (but 변하기도 함)
어미 : 용언의 어간에 붙어 활용하며 변하는 부분, 문법적 기능 수행'''


from nltk.corpus import stopwords
'''4) 불용어

1. 불용어 확인'''

stop_words_list = stopwords.words('english')
print('불용어 개수 :', len(stop_words_list))
print('불용어 10개 출력 :',stop_words_list[:10])

# 2. 불용어 제거

example = "Family is not an important thing. It's everything."
stop_words = set(stopwords.words('english')) 

word_tokens = word_tokenize(example)

result = []
for word in word_tokens: 
    if word not in stop_words: 
        result.append(word) 

print('불용어 제거 전 :',word_tokens) 
print('불용어 제거 후 :',result)


# 5) 정규표현식

import re


# 6) 정수 인코딩

# 1. dictionary 사용

from nltk.tokenize import sent_tokenize

raw_text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."
sentences = sent_tokenize(raw_text) # 문장 토큰화

vocab = {}
preprocessed_sentences = []
stop_words = set(stopwords.words('english')) # 불용어 목록

for sentence in sentences:
    # 단어 토큰화
    tokenized_sentence = word_tokenize(sentence)
    result = []

    for word in tokenized_sentence: 
        word = word.lower() # 단어 소문자화
        if word not in stop_words: # 불용어 제거
            if len(word) > 2: # 단어 길이가 2이하 제거 
                result.append(word)
                if word not in vocab:
                    vocab[word] = 0 
                vocab[word] += 1 # 단어 개수 구하기
    preprocessed_sentences.append(result) 

from nltk import FreqDist
import numpy as np

# np.hstack으로 문장 구분을 제거
vocab = FreqDist(np.hstack(preprocessed_sentences))

vocab_size = 5
vocab = vocab.most_common(vocab_size) # 등장 빈도순

word_to_index = {word[0] : index + 1 for index, word in enumerate(vocab)} # enumerate : 순서가 있는 자료형을 입력으로 받아 인덱스 순차적으로 함께 리턴

print(word_to_index)

# 2. 케라스 텍스트 전처리

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences) # 빈도수 기준 단어 집합 생성

print(tokenizer.word_index) # 인덱스
print(tokenizer.word_counts) # 개수


# 07) 패딩 Padding : 여러 문장의 길이 동일하게 유지

# 1. Numpy 패딩

encoded = tokenizer.texts_to_sequences(preprocessed_sentences) # 인덱스 대입
print(encoded)

max_len = max(len(item) for item in encoded) # 가장 긴 문장의 단어 수

for sentence in encoded:
    while len(sentence) < max_len:
        sentence.append(0)

padded_np = np.array(encoded)
padded_np # 0 채우면 제로 패딩 zero padding

# 2. 케라스 전처리 도구 패딩

from tensorflow.keras.preprocessing.sequence import pad_sequences

padded = pad_sequences(encoded) # 0을 앞에 채움
padded = pad_sequences(encoded, padding='post') # 뒤


''' 8) 원-핫 인코딩 One-Hot Encoding

단어 집합 vocabulary : 서로 다른 단어들의 집합

1. 원-핫 인코딩 : 단어 집합의 크기를 벡터 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고 다른 인덱스에는 0을 부여하는 벡터 표현 방식'''

okt = Okt()  
tokens = okt.morphs("나는 자연어 처리를 배운다")  # 형태소

word_to_index = {word : index for index, word in enumerate(tokens)}

def one_hot_encoding(word, word_to_index):
  one_hot_vector = [0]*(len(word_to_index))
  index = word_to_index[word]
  one_hot_vector[index] = 1
  return one_hot_vector

one_hot_encoding("자연어", word_to_index)

# 한계 : 단어의 유사성 판별 불가


''' 9) 데이터의 분리

1. 지도학습 Supervised Learning

X_train , y_train, X_test, y_train

2. X와 Y 분리

- zip 활용 분리 : 인덱스 0과 인덱스 1 분리'''

s = [['a',1],['b',2],['c',3]]
x, y =zip(*s)
print(x,y)

# - 데이터프레임으로 분리

import pandas as pd

values = [['당신에게 드리는 마지막 혜택!', 1],
['내일 뵐 수 있을지 확인 부탁드...', 0],
['도연씨. 잘 지내시죠? 오랜만입...', 0],
['(광고) AI로 주가를 예측할 수 있다!', 1]]
columns = ['메일 본문', '스팸 메일 유무']

df = pd.DataFrame(values, columns=columns)

X = df['메일 본문']
y = df['스팸 메일 유무']

# - numpy로 분리

np_array = np.arange(0,16).reshape((4,4))

X = np_array[:, :3]
y = np_array[:,3]

# 3. 사이킷 런으로 테스트 데이터 분리

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234) # random_state : 데이터 다른 순서대로 섞임


# 10) 한국어 전처리 패키지

# 1. PyKoSpacing : 띄어쓰기 딥 러닝 모델

# pip install git+https://github.com/haven-jeon/PyKoSpacing.git

'''
from pykospacing import Spacing
spacing = Spacing()

kospacing_sent = spacing(new_sent) 
'''

# 2. Py-Hanspell : 맞춤법 검사 (띄어쓰기 보정 기능도 탑재)

# pip install git+https://github.com/ssut/py-hanspell.git

from hanspell import spell_checker

sent = "맞춤법 틀리면 외 않되? 쓰고싶은대로쓰면돼지 "
spelled_sent = spell_checker.check(sent)

hanspell_sent = spelled_sent.checked
print(hanspell_sent)

# 3. SOYNLP : 단어토큰화 (신조어 문제 해결), 반복되는 문자 정제

# pip install soynlp

# 4. Customised KoNLPy : 사용자 사전 추가 가능
