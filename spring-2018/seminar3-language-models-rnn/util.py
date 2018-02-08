import itertools as it

from nltk.tokenize import word_tokenize


signs = {
    'gemini': 'Близнецы', 'cancer': 'Рак', 'leo': 'Лев', 'libra': 'Весы', 'pisces': 'Рыбы',
    'sagittarius': 'Стрелец', 'scorpio': 'Скорпион', 'taurus': 'Телец', 'virgo': 'Дева',
    'capricorn': 'Козерог', 'aries': 'Овен', 'aquarius': 'Водолей'
}

def read_horoscopes(tokenize=False):
    horoscopes_by_sign = {}
    for sign in signs:
        with open('data/horoscopes/horoscope_{}.txt'.format(sign)) as horoscopes:
            if tokenize:
                horoscopes_by_sign[sign] = [' '.join(word_tokenize(line)) for line in horoscopes.readlines()]
            else:
                horoscopes_by_sign[sign] = horoscopes.readlines()
    unified_horoscopes = it.chain.from_iterable(horoscopes_by_sign.values())
    return '\n'.join(unified_horoscopes)