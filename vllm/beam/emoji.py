from vllm.beam.emoji_data import EMOJI_DATA

_EMOJI_SEARCH_TREE = None

def emoji_count(input: str) -> int:
    return len(emoji_list(input))

def emoji_list(input: str) -> list:
    _entities = []

    def f(emj, emj_data):
        _entities.append({
            'match_start': emj_data['match_start'],
            'match_end': emj_data['match_end'],
            'emoji': emj,
        })

    demojize(input, language='en', version=-1, handle_version=f)
    return _entities

def demojize(
        string,
        delimiters=(":", ":"),
        language='en',
        version=None,
        handle_version=None
):
    """
    Replace unicode emoji in a string with emoji shortcodes. Useful for storage.
        >>> import emoji
        >>> print(emoji.emojize("Python is fun :thumbs_up:"))
        Python is fun 👍
        >>> print(emoji.demojize(u"Python is fun 👍"))
        Python is fun :thumbs_up:
        >>> print(emoji.demojize(u"Unicode is tricky 😯", delimiters=("__", "__")))
        Unicode is tricky __hushed_face__

    :param string: String contains unicode characters. MUST BE UNICODE.
    :param delimiters: (optional) User delimiters other than ``_DEFAULT_DELIMITER``
    :param language: Choose language of emoji name: language code 'es', 'de', etc. or 'alias'
        to use English aliases
    :param version: (optional) Max version. If set to an Emoji Version,
        all emoji above this version will be removed.
    :param handle_version: (optional) Replace the emoji above ``version``
        instead of removing it. handle_version can be either a string or a
        callable ``handle_version(emj: str, data: dict) -> str``; If it is
        a callable, it's passed the unicode emoji and the data dict from
        emoji.EMOJI_DATA and must return a replacement string  to be used.
        The passed data is in the form of::

            handle_version(u'\\U0001F6EB', {
                'en' : ':airplane_departure:',
                'status' : fully_qualified,
                'E' : 1,
                'alias' : [u':flight_departure:'],
                'de': u':abflug:',
                'es': u':avión_despegando:',
                ...
            })

    """
    if language == 'alias':
        language = 'en'
        _use_aliases = True
    else:
        _use_aliases = False

    tree = _get_search_tree()
    result = []
    i = 0
    length = len(string)
    while i < length:
        consumed = False
        char = string[i]
        if char in tree:
            j = i + 1
            sub_tree = tree[char]
            while j < length and string[j] in sub_tree:
                sub_tree = sub_tree[string[j]]
                j += 1
            if 'data' in sub_tree:
                emj_data = sub_tree['data']
                code_points = string[i:j]
                replace_str = None
                if version is not None and emj_data['E'] > version:
                    if callable(handle_version):
                        emj_data = emj_data.copy()
                        emj_data['match_start'] = i
                        emj_data['match_end'] = j
                        replace_str = handle_version(code_points, emj_data)
                    elif handle_version is not None:
                        replace_str = str(handle_version)
                    else:
                        replace_str = None
                elif language in emj_data:
                    if _use_aliases and 'alias' in emj_data:
                        replace_str = delimiters[0] + emj_data['alias'][0][1:-1] + delimiters[1]
                    else:
                        replace_str = delimiters[0] + emj_data[language][1:-1] + delimiters[1]
                else:
                    # The emoji exists, but it is not translated, so we keep the emoji
                    replace_str = code_points

                i = j - 1
                consumed = True
                if replace_str:
                    result.append(replace_str)

        if not consumed and char != u'\ufe0e' and char != u'\ufe0f':
            result.append(char)
        i += 1

    return "".join(result)

def _get_search_tree():
    """
    Generate a search tree for demojize().
    Example of a search tree::

        EMOJI_DATA =
        {'a': {'en': ':Apple:'},
        'b': {'en': ':Bus:'},
        'ba': {'en': ':Bat:'},
        'band': {'en': ':Beatles:'},
        'bandit': {'en': ':Outlaw:'},
        'bank': {'en': ':BankOfEngland:'},
        'bb': {'en': ':BB-gun:'},
        'c': {'en': ':Car:'}}

        _SEARCH_TREE =
        {'a': {'data': {'en': ':Apple:'}},
        'b': {'a': {'data': {'en': ':Bat:'},
                    'n': {'d': {'data': {'en': ':Beatles:'},
                                'i': {'t': {'data': {'en': ':Outlaw:'}}}},
                        'k': {'data': {'en': ':BankOfEngland:'}}}},
            'b': {'data': {'en': ':BB-gun:'}},
            'data': {'en': ':Bus:'}},
        'c': {'data': {'en': ':Car:'}}}

                   _SEARCH_TREE
                 /     |        ⧵
               /       |          ⧵
            a          b             c
            |        / |  ⧵          |
            |       /  |    ⧵        |
        :Apple:   ba  :Bus:  bb     :Car:
                 /  ⧵         |
                /    ⧵        |
              :Bat:    ban     :BB-gun:
                     /     ⧵
                    /       ⧵
                 band       bank
                /   ⧵         |
               /     ⧵        |
            bandi :Beatles:  :BankOfEngland:
               |
            bandit
               |
           :Outlaw:


    """
    global _EMOJI_SEARCH_TREE
    if _EMOJI_SEARCH_TREE is None:
        _EMOJI_SEARCH_TREE = {}
        for emj in EMOJI_DATA:
            sub_tree = _EMOJI_SEARCH_TREE
            lastidx = len(emj) - 1
            for i, char in enumerate(emj):
                if char not in sub_tree:
                    sub_tree[char] = {}
                sub_tree = sub_tree[char]
                if i == lastidx:
                    sub_tree['data'] = EMOJI_DATA[emj]
    return _EMOJI_SEARCH_TREE
