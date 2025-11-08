import sys
import os
import re
import argparse
import copy
import unicodedata

import unittest
import signal
import code
import inspect

# Dump a variable in the current stack for debugging
def dump_var_in_stack(varname):
    for fi in inspect.stack():
        frame = fi.frame
        if varname in frame.f_locals:
            print(f"\nFound '{varname}' in {fi.function} "
                  f"(File {fi.filename}, line {fi.lineno}):")
            print(f"    {varname} = {frame.f_locals[varname]!r}")

def debug_signal_handler(signal, frame):
    code.interact(local=dict(globals(), **locals()))
    print("Debug signal handler invoked. Current local variables:")

# Use command: kill -SIGUSR1 <pid>
signal.signal(signal.SIGUSR1, debug_signal_handler)

# Step 1: Pinyin with diacritics to Toneless Pinyin

"""
Convert Pinyin with diacritics → Toneless Pinyin → Shuangpin (Xiaohe scheme),
with unit tests.
"""

TONELESS_MAPPING = """
ā a 1
á a 2
ǎ a 3
à a 4
ē e 1
é e 2
ě e 3
è e 4
ê ai 0
ê̄ ai 1
ế ai 2
ê̌ ai 3
ề ai 4
ḿ m 2
m̀ m 4
ń n 2
ň n 3
ǹ n 4
ō o 1
ó o 2
ǒ o 3
ò o 4
ī i 1
í i 2
ǐ i 3
ì i 4
ū u 1
ú u 2
ǔ u 3
ù u 4
ü v 0
ǖ v 1
ǘ v 2
ǚ v 3
ǜ v 4
"""

TONES = {
    0: 's',
    1: 'l',
    2: 'r',
    3: 'v',
    4: 'j',
    5: 's'
}

def get_toneless_mapping():
    mapping = dict()
    for line in TONELESS_MAPPING.split('\n'):
        line = line.strip()
        if len(line) == 0:
            continue
        pinyin, toneless, tone  = line.split()
        mapping[pinyin] = (toneless, int(tone))
    return mapping

kTonelessMapping = get_toneless_mapping()

# get the toneless pinyin from pinyin
def get_toneless_pinyin(pinyin):
    toneless = ''
    tone = 0
    for c in unicodedata.normalize('NFC', pinyin):
        if c in kTonelessMapping:
            toneless += kTonelessMapping[c][0]
            tone = kTonelessMapping[c][1]
        elif unicodedata.category(c).startswith('L'):
            # If the character is a letter, keep it as is
            toneless += c
    return (toneless, tone)

# get the toneless pinyin seq from pinyin seq
def get_toneless_pinyin_seq(pinyin_seq):
    return [get_toneless_pinyin(pinyin) for pinyin in pinyin_seq]

# Step 2: Xiaohe Shuangpin mapping (initials + finals)
INITIALS = {
    "b": "b", "p": "p", "m": "m", "f": "f",
    "d": "d", "t": "t", "n": "n", "l": "l",
    "g": "g", "k": "k", "h": "h",
    "j": "j", "q": "q", "x": "x",
    "zh": "v", "ch": "i", "sh": "u", "r": "r",
    "z": "z", "c": "c", "s": "s",
    "y": "y", "w": "w", "": ""
}

FINALS = {
    "iu": "q", "ei": "w", "uan":"r", "van" : "r",  "ue": "t", "ve": "t", "üe": "t",
    "un": "y", "vn": "y", "ün":"y", "uo": "o", "o": "o", "ie": "p",

    "a": "a", "ong": "s", "iong": "s", "ai": "d", "en":"f", "eng": "g",
    "ang": "h", "an": "j", "uai":"k", "ing":"k", "uang": "l", "iang": "l",

    "ou": "z", "ua": "x", "ia": "x", "ao": "c", "ui": "v", "v":"v",
    "in": "b", "iao": "n", "ian": "m",

    "a": "a", "e": "e", "i": "i", "u": "u", "ü":"v"
}

def pinyin_to_shuangpin(toneless_tone):
    """Convert toneless Pinyin syllable to Shuangpin (Xiaohe)."""
    toneless, tone = toneless_tone
    toneless.strip()
    if not toneless:
        raise ValueError("Input Pinyin cannot be empty.")
    special_cases = {"a": "aa", "o": "oo", "e": "ee", "ang": "ah", "eng": "eg",
                     "ng": "eg", "m": "mm", "n": "nn", "hng": "hg",
                     "lü": "lv", "nü": "nv"}
    if toneless in special_cases:
        return special_cases[toneless] + TONES[tone]

    if len(toneless) == 2:
        return toneless + TONES[tone] # Handle two-letter finals directly

    # split initial and final with RMM(right maximum match) algorithm
    i = len(toneless)
    pos = i
    while i > 0:
        i -= 1
        if toneless[i:] in FINALS:
            pos = i
    final = toneless[pos:]
    initial = toneless[:pos]
    assert initial in INITIALS, f"Unknown initial: {initial}, {toneless}"
    assert final in FINALS, f"Unknown final: {final}, {initial}, {toneless}"
    return INITIALS[initial] + FINALS[final] + TONES[tone]

# Step 3: Get standard Chinese characters and Pinyin mappings

STANDARD_CHINESE = "standard_chinese.txt"
PINYIN_CODE = "pinyin.txt"
PINYIN_PHRASE = "pinyin_phrase.txt"

# get chinese code from a file with format "pinyin: word1 word2 ..."
def get_standard_code_from_file(file):
    words = dict()
    # read the file line by line
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            #skip line beginning with '#' or empty line
            if len(line) == 0 or line[0] == '#':
                continue
            # split the line into words by spaces or commas or colons
            chars = re.split(r'[\s,:]', line)
            if len(chars) <= 1:
                continue
            pinyin = chars[0]
            for char in chars[1:]:
                if len(char.strip()) == 0:
                    continue
                if char not in words:
                    words[char] = []
                words[char].append(pinyin)
    return words

# get pinyin code from a file with format "UNICODE: py1,py2 # word"
def get_pinyin_code_from_file(file):
    words = dict()
    with open(file  , 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == '#':
                continue
            parts = re.split(r'\s+', line)
            if len(parts) < 4:
                continue
            word = parts[3].strip()
            pinyins = re.split(r',', parts[1])
            if word not in words:
                words[word] = []
            for pinyin in pinyins:
                words[word].append(pinyin)
    return words

kPinyinCodes = get_pinyin_code_from_file(PINYIN_CODE)
kStandardCodes = get_standard_code_from_file(STANDARD_CHINESE)

# Merge kStandardCodes and kPinyinCodes, then return the new dictionary
def merge_character_codes():
    codes = dict()
    for word in kStandardCodes:
        codes[word] = copy.deepcopy(kStandardCodes[word])
    for word in kPinyinCodes:
        if word not in codes:
            codes[word] = copy.deepcopy(kPinyinCodes[word])
    return codes

kCharacterCodes = merge_character_codes()

# Get pinyin phrase from a file with format "word: py1 py2 ..."
# return a dictionary of word and a list of pinyin code sequences, e.g. {'word': [['py1', 'py2'], ['py3', 'py4']]}
def get_pinyin_phrase_from_file(file):
    words = dict()
    with open(file , 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == '#':
                continue
            parts = re.split(r':\s+', line)
            if len(parts) < 2:
                continue
            word = parts[0].strip()
            pinyins = re.split(r'\s+', parts[1])
            if word not in words:
                words[word] = []
            words[word].append(pinyins)
    return words

# get pinyin phrases
def get_pinyin_phrases():
    return get_pinyin_phrase_from_file(PINYIN_PHRASE)

# Check if a word and its pinyin code sequence are consistent
# word: a Chinese word
# word_code: a list of pinyin codes for the word, e.g. ['py1', 'py2']

def is_consistent(word, word_code, strict=True):
    if len(word) != len(word_code):
        return False
    for i in range(len(word)):
        if word[i] not in kCharacterCodes:
            return False
        if strict and word_code[i] not in kCharacterCodes[word[i]]:
            return False
        if not strict:
            toneless, _ = get_toneless_pinyin(word_code[i])
            if toneless not in [py for py,_ in get_toneless_pinyin_seq(kCharacterCodes[word[i]])]:
                return False
            return True
    return True

# Purge inconsistent phrases
# words: a dictionary of word and a list of pinyin code sequences, e.g. {'word': [['py1', 'py2'], ['py3', 'py4']]}
# strict: if True, check the pinyin with tones; if False, check the toneless pinyin
def purge_inconsistent_phrases(words, strict=True):
    res = dict()
    for word in words.keys():
        for pinyin_seq in words[word]:
            if is_consistent(word, pinyin_seq, strict = strict):
                if word not in res:
                    res[word] = []
                res[word].append(pinyin_seq)
    return res

kPinyinPhrases = purge_inconsistent_phrases(get_pinyin_phrases(), strict=False)

# Step 4: Get frequency-sorted Chinese dictionary

PINYIN_DICT = "pinyin_trad.dict.txt"
PINYIN_EXT1_DICT = "pinyin_trad_ext1.dict.txt"

# get the frequency of words from a file with format "word<tab>code<tab>frequency"
def get_frequency_from_files(files):
    freq = dict()
    for file in files:
        line_no = 0
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                line_no += 1
                word, code, frequency = line.split('\t')
                code = re.sub(r'(?<=[ln])ue', 've', code)
                if word not in freq:
                    freq[word] = dict()
                if code not in freq[word]:
                    freq[word][code] = (0, 0)  # (frequency, placeholder for future use)
                f, p = freq[word][code]
                freq[word][code] = (f + int(frequency), -line_no if p == 0 else p)
    return freq

kWordsFreq = get_frequency_from_files([PINYIN_DICT, PINYIN_EXT1_DICT])

# Get toneless pinyin phrases from kPinyinPhrases and kWordsFreq
def get_toneless_pinyin_phrases():
    toneless_phrases = dict()
    for word in kPinyinPhrases:
        for pinyin_seq in kPinyinPhrases[word]:
            toneless_seq = get_toneless_pinyin_seq(pinyin_seq)
            toneless_code = ' '.join([py for py, _ in toneless_seq])
            if word not in toneless_phrases:
                toneless_phrases[word] = set()
            toneless_phrases[word].add(toneless_code)
    for word in kWordsFreq:
        if len(word) <= 1:
            continue
        for toneless_code in kWordsFreq[word]:
            if word not in toneless_phrases:
                toneless_phrases[word] = set()
            toneless_phrases[word].add(toneless_code)
    # convert set to list
    for word in toneless_phrases:
        toneless_phrases[word] = [toneless_code.split() for toneless_code in toneless_phrases[word]]
    return toneless_phrases

kTonelessPinyinPhrases = get_toneless_pinyin_phrases()

# get the frequency of a character from freq_dict with default value 0.
def get_freq_of_word(word, toneless_code, freq_dict):
    if word not in freq_dict:
        return (0, -sys.maxsize + 1)
    if toneless_code not in freq_dict[word]:
        return (0, -sys.maxsize + 1)
    return freq_dict[word][toneless_code]

# Step 5: Get Cangjie codes from file

CANGJIE_CODE = "cangjie5.dict.yaml"

# get cangjie code from a file with format "word\tcjcode1 extra_info"
def get_cangjie_code_from_file(file):
    words = dict()
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == '#':
                continue
            parts = re.split(r'\t', line)
            if len(parts) < 2:
                continue
            word = parts[0].strip()
            cjcode = parts[1].strip().split()[0]
            # skip cjcode prefixed with 'x' or 'z'
            if cjcode[0] in ['x', 'z']:
                continue
            if word not in words:
                words[word] = []
            words[word].append(cjcode)
    return words

kCangjieCodes = get_cangjie_code_from_file(CANGJIE_CODE)

# Step 6: Main function to convert toneless pinyin to shuangpin

def pinyin_to_shuangpin_seq(toneless_seq):
    """Convert a sequence of toneless Pinyin to Shuangpin (Xiaohe scheme)."""
    shuangpin_seq = []
    for toneless in toneless_seq:
        shuangpin = pinyin_to_shuangpin(toneless)
        shuangpin_seq.append(shuangpin)
    return shuangpin_seq

# Get all descartes products of encodes which is a list of list of elements
# e.g. [[a1, a2], [b1, b2]] -> [[a1, b1], [a1, b2], [a2, b1], [a2, b2]]
def get_descartes_products(encodes):
    descartes = [[]]
    for encode in encodes:
        new_descartes = []
        for descarte in descartes:
            for element in encode:
                new_descartes.append(descarte + [element])
        descartes = new_descartes
    return descartes

# Get Cangjie quick5 codes for a word
def get_cangjie_quick5(word):
    codes = []
    for c in word:
        cjcodes = [''.join([cjcode[0], cjcode[-1]]) for cjcode in kCangjieCodes.get(c, [])]
        if len(cjcodes) == 0:
            raise ValueError(f"No Cangjie codes found for character '{c}' in word '{word}'.")
        codes.append(cjcodes)
    codes = [''.join(code) for code in get_descartes_products(codes)]
    return codes

# Get initial or final codes of Cangjie for a word
def get_initial_or_finals_cangjie5(word, mode):
    codes = []
    if mode in ['first-last']:
        codes = [cjcode[0] + cjcode[-1] for cjcode in get_cangjie_quick5(word)]
    elif mode in ['last-first']:
        codes = [cjcode[-1] + cjcode[0] for cjcode in get_cangjie_quick5(word)]
    elif mode in ['first']:
        codes = [cjcode[0] for cjcode in get_cangjie_quick5(word)]
    elif mode in ['last']:
        codes = [cjcode[-1] for cjcode in get_cangjie_quick5(word)]
    else:
        assert mode in ['none']
        codes = ['']
    if len(codes) == 0:
        raise ValueError(f"No Cangjie codes found for word '{word}', with mode '{mode}'.")
    return codes

# Return a list of FlypyQuick5 sequences for a word and its Pinyin sequence.
# This function converts the Pinyin sequence to toneless Pinyin,
# then maps each toneless Pinyin to its Flypys, and then to its corresponding FlypyQuick5 code.
# The mapping is based on the Xiaohe Shuangpin scheme.

# The FlypyQuick5 sequences are generated based on the following rules:
# Notations: flypys is a list of Flypys for the toneless Pinyin,
#          cjcode is the Cangjie code for the last Chinese character in word.
# Rules:
# 1. <= 3 characters: "".join(flypys) + cjcode[0] + cjcode[-1]
# 3. >= 4 characters: "".join(flypys)
# 5. If the word is not in the frequency dictionary, use a default frequency of

# Returns a list of FlypyQuick5 sequences for the given word and Pinyin sequence.
#  [("flypyquick5_seq1", freq1), ("flypyquick5_seq2", freq2), ...]

def get_frequency_default(word, code):
    return get_freq_of_word(word, code, kWordsFreq)

def get_flypyquick5_seq(word, pinyin_seq, get_frequency = get_frequency_default):
    """Convert a word and its Pinyin sequence to FlypyQuick5 sequences."""
    try:
        toneless_seq = get_toneless_pinyin_seq(pinyin_seq)
        flypys = pinyin_to_shuangpin_seq(toneless_seq)
    except ValueError as e:
        raise ValueError(f"Error converting Pinyin to Shuangpin for word '{word}', '{pinyin_seq}'")
    freq = get_frequency(word, ' '.join([py for py, _ in toneless_seq]))

    # mode
    assert len(word) >= 1
    mode = 'none'
    if len(word) <= 3:
        mode = 'last-first'

    pys = ''.join(flypys)
    flypyquick5_seq = [(''.join(code), freq) for code in get_descartes_products([[pys], get_initial_or_finals_cangjie5(word, mode)])]
    if len(flypyquick5_seq) == 0:
        raise ValueError(f"No valid FlypyQuick5 sequences generated for word '{word}'.")
    # Return the list of FlypyQuick5 sequences
    return flypyquick5_seq

# Get the FlypyQuick5 dictionary from a dictionary of words and their Pinyin sequences.
# words: a dictionary of word and a list of pinyin code sequences, e.g. {'word': [['py1', 'py2'], ['py3', 'py4']]}
# return a dictionary of word and a list of FlypyQuick5 sequences, e.g. {'word': [("flypyquick5_seq1", freq1), ("flypyquick5_seq2", freq2), ...]}

def get_flypyquick5_dict(words, get_frequency = get_frequency_default):
    flypyquick5_dict = dict()
    for word in words.keys():
        for pinyin_seq in words[word]:
            try:
                flypyquick5_seq = get_flypyquick5_seq(word, pinyin_seq, get_frequency)
                if word not in flypyquick5_dict:
                    flypyquick5_dict[word] = []
                for seq, freq in flypyquick5_seq:
                    flypyquick5_dict[word].append((seq, freq))
            except ValueError as e:
                print(f"Warning: {e}", file=sys.stderr)
    return flypyquick5_dict

# Get pinyin sequences for words from a dictionary of words and their pinyin code sequences. If a word is not in the dictionary, return a descartes product of its characters' pinyin codes.
# words is a list of words, e.g. ['word1', 'word2', ...]
# return a dictionary of word and a list of pinyin code sequences, e.g. {'word': [['py1', 'py2'], ['py3', 'py4']]}

def get_pinyin_seq_for_words(words):
    pinyin_seq_dict = dict()
    for word in words:
        if word in kPinyinPhrases:
            pinyin_seq_dict[word] = kPinyinPhrases[word]
        else:
            encodes = []
            for char in word:
                if char in kCharacterCodes:
                    encodes.append(kCharacterCodes[char])
                else:
                    encodes.append([''])
            pinyin_seq_dict[word] = get_descartes_products(encodes)
    return pinyin_seq_dict

# Get a list of words from an input file, one word per line.
def get_words_from_file(file, words = dict(), min_length=2, max_length=7):
    lineno = 0
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            lineno += 1
            if len(line) == 0:
                continue
            if line[0] == '#':
                continue
            items = line.split()
            word = items[0]
            frequency = 0
            if len(items) >= 2:
                try:
                    frequency = int(items[-1])
                except ValueError:
                    pass
            if len(word) < min_length or len(word) > max_length:
                continue
            if word not in words:
                words[word] = (0, 0)
            f, p = words[word]
            words[word] = (f + frequency, -lineno if p == 0 else p)
    return words

# Get the difference set of phrases against the builtin Pinyin phrases.
def get_difference_set(phrase_list):
    """Get the difference set of phrases against the builtin Pinyin phrases."""
    diff_list = []
    for word in phrase_list:
        if word not in kTonelessPinyinPhrases:
            diff_list.append(word)
    return diff_list

# Step 7: Header for Rime dictionary
def get_header(name, input_tables = []):
    hdr = f"""# rime dictionary
# encoding: utf-8

---
name: {name}
version: "0.1"
sort: by_weight"""

    if input_tables:
        hdr += """
max_phrase_length: 10
min_phrase_weight: 100
encoder:
  exclude_patterns:
    - '^[a-z].?$'
    - '^[a-z]..$'
  rules:
    - length_equal: 2
      formula: "AaAbBaBbBcAd"
    - length_equal: 3
      formula: "AaAbBaBbCaCbCcAd"
    - length_equal: 4
      formula: "AaAbBaBbCaCbDaDb"
    - length_equal: 5
      formula: "AaAbBaBbCaCbDaDbEaEb"
    - length_equal: 6
      formula: "AaAbBaBbCaCbDaDbEaEbFaFb"
    - length_equal: 7
      formula: "AaAbBaBbCaCbDaDbEaEbFaFbGaGb"
import_tables:"""
        for table in input_tables:
            hdr += f"\n  - {table}"
    hdr += "\n...\n"
    return hdr

# get the sorted keys of a dictionary
def get_sorted_keys(dict):
    keys = list(dict.keys())
    keys.sort()
    return keys

# Convert a dictionary of word codes to a nested list format.
# word_codes: a dictionary of word and a list of tonal pinyin code sequences,
#               e.g. {'word': ["py1", "py2", ...]}]}
# return a nested list
#        e.g. {"word": [["py1"], ["py2"], ...]}
def convert_to_nested_dict(word_codes):
    nested_dict = dict()
    for word in word_codes:
        if word not in nested_dict:
            nested_dict[word] = []
        for code in word_codes[word]:
            nested_dict[word].append([code])
    return nested_dict

# Get the list of word tuple (freq, code, word) from a nested dictionary of length, code, word and frequency
# sorted_dict: a nested dictionary of length, code, word and
# word_codes: a nested dictionary of length, code, word and frequency
# return a list of word tuple (freq, code, word) in descending order of frequency
def get_sorted_word_tuples(sorted_dict):
    word_tuples = []
    for length in sorted_dict:
        for code in sorted_dict[length]:
            for word in sorted_dict[length][code]:
                freq = sorted_dict[length][code][word]
                word_tuples.append((freq, code, word))
    word_tuples.sort(reverse=True)
    return word_tuples

# print the word_codes which is a dictionary of key,list into a file with the format of word code frequency
# word_codes: a dictionary of word and a list of tonal pinyin code sequences,
#               e.g. {'word': [("py1 py2", freq),  ("py3 py4", freq), ...]}
# return a nested dictionary of length, code, word and frequency
#        e.g. {"length": {"code": {"word": frequency}}}

def sort_by_length_and_code(word_codes):
    sorted_word_codes = dict()
    for word in word_codes:
        length = len(word)
        if length not in sorted_word_codes:
            sorted_word_codes[length] = dict()
        for code, freq in word_codes[word]:
            if code not in sorted_word_codes[length]:
                sorted_word_codes[length][code] = dict()
            if word not in sorted_word_codes[length][code]:
                sorted_word_codes[length][code][word] = (0, 0)  # (frequency, placeholder for future use)
            f, p = sorted_word_codes[length][code][word]
            sorted_word_codes[length][code][word] = (max(f, freq[0]), freq[1] if p == 0 else p)
    return sorted_word_codes

# print the word_codes which is a dictionary of key,list into a file with the format of word code frequency
# word_codes: {"legth": {"code": {"word": frequency}}}
# outfile: the output file, default is sys.stdout, in the following format:
# word<tab>code<tab>frequency
def print_word_codes(word_codes, outfile=sys.stdout, freq_base=0, max_length=7):
    for length in get_sorted_keys(word_codes):
        for code in get_sorted_keys(word_codes[length]):
            for word in get_sorted_keys(word_codes[length][code]):
                if len(word) > max_length:
                    continue
                freq = word_codes[length][code][word]
                print("%s\t%s\t%i" % (word, code, freq[0]+freq_base), file=outfile)

# Get the sorted FlypyQuick5 dictionary from a list of words.
# words: a dictionary of words, {"word": [["py1", "py2"], ["py3", "py4"]]}
# return a nested dictionary of length, code, word and frequency
def get_sorted_flypyquick5_dict(words, get_frequency = get_frequency_default):
    words_dict = get_flypyquick5_dict(words, get_frequency)
    sorted_dict = sort_by_length_and_code(words_dict)
    return sorted_dict

# Obsolete function
# Augment the common words when there are conflicts by appending the first character's Cangjie code to the FlypyQuick5 code.
# which are not most frequent ones.
# word_codes: a nested dictionary of length, code, word and frequency
def augment_common_words(word_codes, builtin_dicts = [dict()], lengths = [1, 2, 3], preemptive=True):
    builtin_codes = dict()
    for builtin_dict in builtin_dicts:
        for length in builtin_dict:
            for code in builtin_dict[length]:
                for word in builtin_dict[length][code]:
                    assert code not in builtin_codes or len(code) > 7, f"Duplicate builtin code {code} for word {word}"
                    if code not in builtin_codes:
                        builtin_codes[code] = (builtin_dict[length][code][word], word)
                    else:
                        builtin_codes[code] = max(builtin_codes[code], (builtin_dict[length][code][word], word))

    codes_to_remove = dict()
    for length in lengths:
        if length not in word_codes:
            continue

        words_to_remove = dict()
        for code in list(word_codes[length].keys()):
            assert len(word_codes[length][code]) > 0, f"Empty code entry for code {code} in length {length}"
            # find the most frequent word by sorting the word tuples
            word_tuples = get_sorted_word_tuples({length: {code: word_codes[length][code]}})
            max_freq = (0, -sys.maxsize)
            max_word = None
            not_in_builtin = code not in builtin_codes
            if not_in_builtin:
                for freq, code, word in word_tuples:
                    codep = code[:-1]
                    if (freq > max_freq) and (len(word_tuples) == 1 or (codep not in builtin_codes or word != builtin_codes[codep][1])):
                        max_freq = freq
                        max_word = word
            assert max_word is not None or not_in_builtin == False, f"max_word is None for code {code} in length {length}"

            # augment other words
            i = 0
            for freq, code, word in word_tuples:
                if word == max_word:
                    continue
                try:
                    j = 0
                    while j < 27:
                        aug_suffix = chr(ord('a') + i % 26)  # 'a', 'b', 'c', ...
                        new_code = code + aug_suffix
                        if (new_code not in word_codes[length]) and (new_code not in builtin_codes or (preemptive and freq > builtin_codes[new_code][0]) or (j == 26 and len(word) == 1)):
                            if new_code in builtin_codes:
                                codes_to_remove[new_code] = builtin_codes[new_code][1]
                            break
                        i += 1
                        j += 1
                    if new_code not in word_codes[length]:
                        word_codes[length][new_code] = dict()
                    if word not in word_codes[length][new_code]:
                        word_codes[length][new_code][word] = (0, 0)
                    f, p = word_codes[length][new_code][word]
                    word_codes[length][new_code][word] = (max(f, freq[0]), freq[1] if p == 0 else p)
                except ValueError as e:
                    print(f"Warning: {e}", file=sys.stderr)
                i += 1
            # remove the old code entry except the most frequent one
            words_to_remove[code] = [word for word in word_codes[length][code] if word != max_word]

        # remove the old code entries
        for code in words_to_remove:
            for word in words_to_remove[code]:
                del word_codes[length][code][word]
            if len(word_codes[length][code]) == 0:
                del word_codes[length][code]

    # remove the codes in codes_to_remove
    for code in codes_to_remove:
        for builtin_dict in builtin_dicts:
            length = len(code)
            if length in builtin_dict and code in builtin_dict[length]:
                if codes_to_remove[code] in builtin_dict[length][code]:
                    del builtin_dict[length][code][codes_to_remove[code]]
                if len(builtin_dict[length][code]) == 0:
                    del builtin_dict[length][code]
    return word_codes

# process a list of words and print the FlypyQuick5 dictionary to a file
# words: a list of words
# outfile: the output file, default is sys.stdout
def process_and_print_flypyquick5_dict(words, outfile=sys.stdout, primary_set = [dict()], freq_base=0, get_frequency = get_frequency_default):
    sorted_dict = get_sorted_flypyquick5_dict(words, get_frequency)
    augmented_dict = augment_common_words(sorted_dict, primary_set)
    print_word_codes(augmented_dict, outfile, freq_base)

# Step 8: Simplified codes for codes of most frequent words

# Get abbreviated codes for the most frequent words
# code_size: the size of the abbreviated code, which is obtained by truncating the FlypyQuick5 code
# word_tuples: a list of word tuple (freq, code, word) in descending order of frequency
# return a dictionary of word and its abbreviated code, which is a nested dictionary of length(code_size), code, word and frequency.
# Only the most frequent word for each abbreviated code is kept.
def get_abbreviated_codes(code_size, word_tuples, used_codes = set(), min_freq=0):
    abbreviated_dict = dict()
    for freq, code, word in word_tuples:
        if freq[0] < min_freq:
            continue
        simple_code = code[:code_size]
        if simple_code in used_codes:
            continue
        used_codes.add(simple_code)
        if len(code) <= code_size:
            continue
        length = code_size
        if length not in abbreviated_dict:
            abbreviated_dict[length] = dict()
        if simple_code not in abbreviated_dict[length]:
            abbreviated_dict[length][simple_code] = dict()
        abbreviated_dict[length][simple_code][word] = freq
    return abbreviated_dict

# Get the simpflified FlypyQuick5 dictionary for the builtin characters and phrases.
# The abbreviated code size is 1 and 2 for the most frequent Chinese characters only, 3 for the most frequent two-character phrases only.
# return a list of nested dictionaries of length(code_size), code, word and frequency.
def get_abbreviated_dict_for(toneless_phrases, characters, used_codes = set(), min_freq=0):
    for length in [2, 3]:
        assert length in toneless_phrases, f"Length {length} not in toneless_phrases"
        for code in toneless_phrases[length]:
            used_codes.add(code)

    # the list of phrase levels to process, each item is a tuple of (phrases_dict, code_sizes)
    phrase_levels = [(characters, [1, 2, 3]), # single characters, 1 and 2-letter codes
                     ({1: characters[1], 2: toneless_phrases[2]}, [4]), # two-character phrases, 4-letter codes
                     ({2: toneless_phrases[2]}, [5]), # two-character phrases, 5-letter codes
                     ({2: toneless_phrases[2]}, [6]), # two-character phrases, 5-letter codes
                     ({2: toneless_phrases[2]}, [7]), # two-character phrases, 5-letter codes
                     ({3: toneless_phrases[3]}, [8]), # three-character phrases, 6-letter codes
                     ({3: toneless_phrases[3]}, [9]), # three-character phrases, 7-letter codes
                     ({3: toneless_phrases[3]}, [7])] # three-character phrases, 5-letter codes

    abbreviated_dicts = []
    for phrases_dict, code_sizes in phrase_levels:
        phrase_tuples = get_sorted_word_tuples(phrases_dict)
        for code_size in code_sizes:
            abbreviated_dicts.append(get_abbreviated_codes(code_size, phrase_tuples, used_codes, min_freq))

    # return the abbreviated dictionary
    return abbreviated_dicts

def append_used_codes(used_codes, word_code_dicts = [dict()]):
    for word_code_dict in word_code_dicts:
        for length in word_code_dict:
            for code in word_code_dict[length]:
                used_codes.add(code)

# ---------------------- Unit Tests ----------------------
class TestShuangpin(unittest.TestCase):
    def test_basic_cases(self):
        self.assertEqual(pinyin_to_shuangpin(("ao", 1)), "aol")
        self.assertEqual(pinyin_to_shuangpin(("fei", 2)), "fwr")
        self.assertEqual(pinyin_to_shuangpin(("ei", 3)), "eiv")
        self.assertEqual(pinyin_to_shuangpin(("ang", 4)), "ahj")
        self.assertEqual(pinyin_to_shuangpin(("eng", 5)), "egs")

    def test_with_initials(self):
        self.assertEqual(pinyin_to_shuangpin(("mao", 1)), "mcl")
        self.assertEqual(pinyin_to_shuangpin(("zhong", 2)), "vsr")
        self.assertEqual(pinyin_to_shuangpin(("xue", 3)), ("xtv"))
        self.assertEqual(pinyin_to_shuangpin(("yu", 4)), "yuj")

    def test_pinyin_to_shuangpin(self):
        self.assertEqual(pinyin_to_shuangpin(("a", 1)), "aal")
        self.assertEqual(pinyin_to_shuangpin(("o", 2)), "oor")
        self.assertEqual(pinyin_to_shuangpin(("e", 3)), "eev")

        self.assertEqual(pinyin_to_shuangpin(("n", 1)), "nnl")
        self.assertEqual(pinyin_to_shuangpin(("m", 2)), "mmr")
        self.assertEqual(pinyin_to_shuangpin(("hng",3)), "hgv")
        self.assertEqual(pinyin_to_shuangpin(("ng", 4)), "egj")

    def test_toneless_conversion(self):
        self.assertEqual(get_toneless_pinyin("mǎ"), ("ma", 3))
        self.assertEqual(get_toneless_pinyin("nǐ"), ("ni", 3))
        self.assertEqual(get_toneless_pinyin("hǎo"), ("hao", 3))
        self.assertEqual(get_toneless_pinyin("lǜ"), ("lv",4))
        self.assertEqual(get_toneless_pinyin("aī"), ("ai", 1))

    def test_toneless_sequence(self):
        pinyin_seq = ["mǎ", "nǐ", "hǎo", "lǜ"]
        toneless_seq = get_toneless_pinyin_seq(pinyin_seq)
        self.assertEqual(toneless_seq, [("ma",3), ("ni", 3), ("hao", 3), ("lv", 4)])

    def test_get_flypyquick5_seq(self):
        testcases = [("略", ["lüè"], "ltjrw"),
                     ("略", ["lue"], "ltsrw"),
                     ("略", ["lve"], "ltsrw"),
                     ("你好", ["nǐ", "hǎo"], "nivhcvdo"),
                     ("长臂猿", ["cháng", "bì", "yuán"], "ihrbijyrrvp"),
                     ("世界地圖", ["shì", "jiè", "dì", "tú"], "uijjpjdijtur"),
                     ("中華人民共和國", ["zhōng", "huá", "rén", "mín", "gòng", "hé", "guó"], "vslhxrrfrmbrgsjhergor")]

        for word, pinyin_seq, expected_seq in testcases:
            flypyquick5_seq = get_flypyquick5_seq(word, pinyin_seq)
            self.assertTrue(len(flypyquick5_seq) > 0)
            for seq, freq in flypyquick5_seq:
                print(f"Word: {word}, Pinyin: {pinyin_seq}, FlypyQuick5: {seq}, Frequency: {freq}")
                self.assertTrue(isinstance(seq, str))
                self.assertTrue(isinstance(freq, tuple))
                self.assertEqual(seq, expected_seq)

    def test_flypyquick5_dict(self):
        words = {"你好": [["nǐ", "hǎo"]], "世界": [["shì", "jiè"]]}
        flypyquick5_dict = get_flypyquick5_dict(words)
        self.assertTrue("你好" in flypyquick5_dict)
        self.assertTrue("世界" in flypyquick5_dict)
        self.assertTrue(len(flypyquick5_dict["你好"]) > 0)
        self.assertTrue(len(flypyquick5_dict["世界"]) > 0)

    def test_pinyin_phrase(self):
        pinyin_phrases = get_pinyin_phrases()
        self.assertTrue("你好" in pinyin_phrases)
        self.assertTrue(len(pinyin_phrases["你好"]) > 0)
        for phrase in pinyin_phrases["你好"]:
            self.assertTrue(isinstance(phrase, list))
            self.assertTrue(len(phrase) > 0)

    def test_is_consistent(self):
        self.assertTrue(is_consistent("你好", ["ni", "hao"], strict=False))
        self.assertFalse(is_consistent("你好", ["ni", "hao", "shi"]))
        self.assertTrue(is_consistent("世界", ["shi", "jie"], strict=False))
        self.assertFalse(is_consistent("世界", ["shi", "jie"], strict=True))

    def test_purge_inconsistent_phrases(self):
        phrases = {
            "你好": [["ni", "hao"], ["ni", "hao", "shi"]],
            "世界": [["shi", "jie"]]
        }
        purged_phrases = purge_inconsistent_phrases(phrases, strict=False)
        self.assertTrue("你好" in purged_phrases)
        self.assertTrue("世界" in purged_phrases)
        self.assertEqual(len(purged_phrases["你好"]), 1)
        self.assertEqual(len(purged_phrases["世界"]), 1)

    def test_kCharacterCodes(self):
        self.assertTrue("你" in kCharacterCodes)
        self.assertTrue("好" in kCharacterCodes)
        self.assertTrue(len(kCharacterCodes["你"]) > 0)
        self.assertTrue(len(kCharacterCodes["好"]) > 0)
        self.assertTrue(isinstance(kCharacterCodes["你"][0], str))
        self.assertTrue(isinstance(kCharacterCodes["好"][0], str))

    def test_kPinyinPhrases(self):
        self.assertTrue("你好" in kPinyinPhrases)
        self.assertTrue(len(kPinyinPhrases["你好"]) > 0)
        for phrase in kPinyinPhrases["你好"]:
            self.assertTrue(isinstance(phrase, list))
            self.assertTrue(len(phrase) > 0)

    def test_convert_to_nested_dict(self):
        word_codes = {
            "好": ["hao3", "hao4"]
        }
        nested_dict = convert_to_nested_dict(word_codes)
        self.assertTrue("好" in nested_dict)
        self.assertTrue(len(nested_dict["好"]) > 0)
        self.assertTrue(isinstance(nested_dict["好"][0], list))
        self.assertEqual(nested_dict["好"][0][0], "hao3")

    def test_difference_set(self):
        phrase_list = ["你好", "世界", "再见", "應該沒有這個詞"]
        diff_list = get_difference_set(phrase_list)
        self.assertTrue("再见" in diff_list)
        self.assertFalse("你好" in diff_list)
        self.assertFalse("世界" in diff_list)
        self.assertTrue("應該沒有這個詞" in diff_list)

    def test_sort_by_length_and_code(self):
        word_codes = {
            "你好": [("nihcd", (100, -1))],
            "世界": [("uijcd", (200, -1))],
            "再见": [("zajd", (150, -1))]
        }
        sorted_dict = sort_by_length_and_code(word_codes)
        self.assertTrue(2 in sorted_dict)
        self.assertTrue("nihcd" in sorted_dict[2])
        self.assertTrue("uijcd" in sorted_dict[2])
        self.assertTrue("zajd" in sorted_dict[2])
        self.assertEqual(sorted_dict[2]["nihcd"]["你好"], (100, -1))
        self.assertEqual(sorted_dict[2]["uijcd"]["世界"], (200, -1))
        self.assertEqual(sorted_dict[2]["zajd"]["再见"], (150, -1))

    def test_augment_common_words(self):
        word_codes = {
            2: {
                "nihcd": {"你好": (100, -1), "你号": (50, -2)},
                "uijpl": {"世界": (200, -1)},
                "juch": {"颶": (100, -1), "犋": (50, -2)},
            }
        }
        augmented_dict = augment_common_words(word_codes)
        self.assertTrue("nihcd" in augmented_dict[2])
        self.assertTrue("uijpl" in augmented_dict[2])
        self.assertTrue("nihcda" in augmented_dict[2])
        self.assertTrue("juch" in augmented_dict[2])
        self.assertTrue("jucha" in augmented_dict[2])
        self.assertEqual(augmented_dict[2]["nihcd"]["你好"], (100, -1))
        self.assertEqual(augmented_dict[2]["nihcda"]["你号"], (50, -2))
        self.assertEqual(augmented_dict[2]["uijpl"]["世界"], (200, -1))
        self.assertEqual(augmented_dict[2]["juch"]["颶"], (100, -1))
        self.assertEqual(augmented_dict[2]["jucha"]["犋"], (50, -2))
        self.assertFalse("nihcd" in augmented_dict[2] and "你号" in augmented_dict[2]["nihcd"])
        self.assertFalse("juch" in augmented_dict[2] and "犋" in augmented_dict[2]["juch"])

    def test_augment_common_words_with_primary(self):
        primary_dict = {
            2: {
                "nihcd": {"你好": (100, -1)},
                "uijcd": {"世界": (200, -1)},
            }
        }
        word_codes = {
            2: {
                "nihcd": {"你好": (100, -1), "你号": (50, -2)},
                "uijpl": {"世界": (200, -1)},
                "juch": {"颶": (100, -1), "犋": (50, -2)},
            }
        }
        augmented_dict = augment_common_words(word_codes, [primary_dict])
        self.assertFalse("nihcd" in augmented_dict[2])
        self.assertTrue("uijpl" in augmented_dict[2])
        self.assertTrue("nihcda" in augmented_dict[2])
        self.assertTrue("nihcdb" in augmented_dict[2])
        self.assertTrue("juch" in augmented_dict[2])
        self.assertTrue("jucha" in augmented_dict[2])
        self.assertEqual(augmented_dict[2]["nihcda"]["你好"], (100, -1))
        self.assertEqual(augmented_dict[2]["nihcdb"]["你号"], (50, -2))
        self.assertEqual(augmented_dict[2]["uijpl"]["世界"], (200, -1))
        self.assertEqual(augmented_dict[2]["juch"]["颶"], (100, -1))
        self.assertEqual(augmented_dict[2]["jucha"]["犋"], (50, -2))
        self.assertFalse("nihcd" in augmented_dict[2] and "你号" in augmented_dict[2]["nihcd"])
        self.assertFalse("juch" in augmented_dict[2] and "犋" in augmented_dict[2]["juch"])

    def test_process_and_print_flypyquick5_dict(self):
        words = {
            "你好": [["nǐ", "hǎo"]],
            "世界": [["shì", "jiè"]]
        }
        from io import StringIO
        output = StringIO()
        process_and_print_flypyquick5_dict(words, outfile=output)
        output_str = output.getvalue()
        self.assertIn("你好", output_str)
        self.assertIn("世界", output_str)
        self.assertIn("nivhcvd", output_str)
        self.assertIn("uijjpjl", output_str)

    def test_get_sorted_word_tuples(self):
        word_codes = {
            2: {
                "nihcd": {"你好": (100, -1)},
                "uijcd": {"世界": (200, -1)},
            }
        }
        word_tuples = get_sorted_word_tuples(word_codes)
        self.assertEqual(len(word_tuples), 2)
        self.assertEqual(word_tuples[0], ((200, -1), "uijcd", "世界"))
        self.assertEqual(word_tuples[1], ((100, -1), "nihcd", "你好"))

    def test_get_abbreviated_codes(self):
        word_tuples = [
            ((200, -1), "uijcd", "世界"),
            ((100, -1), "nihcd", "你好"),
            ((50, -1), "nihcdo", "你号")
        ]
        abbreviated_dict = get_abbreviated_codes(4, word_tuples)
        self.assertTrue(4 in abbreviated_dict)
        self.assertTrue("uijc" in abbreviated_dict[4])
        self.assertTrue("nihc" in abbreviated_dict[4])
        self.assertEqual(abbreviated_dict[4]["uijc"]["世界"], (200, -1))
        self.assertEqual(abbreviated_dict[4]["nihc"]["你好"], (100, -1))
        self.assertFalse("nihc" in abbreviated_dict[4] and "你号" in abbreviated_dict[4]["nihc"])

# Steo 8: Command-line interface
def main():
    parser = argparse.ArgumentParser(description="Convert Pinyin with diacritics to Shuangpin (Xiaohe scheme).")
    parser.add_argument("--name", help="Name of the current tables", required = False)
    parser.add_argument("--chinese_code", help="Print chinese code into file", action="store_true")
    parser.add_argument("--phrase", help="Print pinyin phrase into file", action="store_true")
    parser.add_argument("--abbreviate", help="Print abbreviated codes into file", action="store_true")
    parser.add_argument("--extra_table", help="Print extra words into file", action="store_true")
    parser.add_argument("--difference", help="Difference the set against the builtin phrases", action="store_true")
    parser.add_argument("--test", help="Run unit tests", action="store_true")
    parser.add_argument("input_files", nargs='*', help="The list of extra input files", default=[])
    args = parser.parse_args()

    args.name = args.name.strip() if args.name else ''

    phrase_suffix = "_phrase"
    abbrev_suffix = "_abbrev"
    extra_suffix = "_extra"
    abbrevextra_suffix = "_abbrevextra"
    file_suffix = ".dict.yaml"
    path = "../"
    input_tables = [args.name + t for t in [phrase_suffix, abbrev_suffix, abbrevextra_suffix]]

    if not args.test:
        tonal_phrases = get_sorted_flypyquick5_dict(kPinyinPhrases)
        characters = get_sorted_flypyquick5_dict(convert_to_nested_dict(kCharacterCodes))
        used_codes = set()
        # Abbreviate codes for the most frequent words
        abbreviated_dicts = get_abbreviated_dict_for(tonal_phrases, characters, used_codes)
        # Augment characters
        augmented_characters = augment_common_words(characters, abbreviated_dicts)
        # Augment phrases
        augmented_phrases = augment_common_words(tonal_phrases, abbreviated_dicts)

    if args.phrase:
        name = args.name + phrase_suffix
        filename = path + name + file_suffix
        with open(filename, 'w', encoding='utf-8') as f:
            # Print Pinyin phrases
            print(get_header(name), file=f)
            print_word_codes(augmented_phrases, f)
        print(f"Pinyin phrases written to {name + file_suffix}")

    if args.abbreviate:
        name = args.name + abbrev_suffix
        filename = path + name + file_suffix
        with open(filename, 'w', encoding='utf-8') as f:
            # Print abbreviated codes for most frequent words
            print(get_header(name), file=f)
            for abbreviated_dict in abbreviated_dicts:
                print_word_codes(abbreviated_dict, f)
        print(f"Abbreviated codes written to {name + file_suffix}")

    if args.input_files and args.extra_table:
        words = dict()
        for input_file in args.input_files:
            words = get_words_from_file(input_file, words)
        if args.difference:
            for w in kTonelessPinyinPhrases.keys():
                if w in words:
                    del words[w]
        print(f"Total {len(words)} extra words read from input files.")
        extra_dict = get_pinyin_seq_for_words(words)

        get_frequency = lambda word, code: words[word] if word in words else (0, -sys.maxsize+1)
        sorted_extra_dict = get_sorted_flypyquick5_dict(extra_dict, get_frequency)

        # Abbreviate codes for the most frequent extra words
        append_used_codes(used_codes, [augmented_characters, augmented_phrases])
        abbreviated_extra_dicts = get_abbreviated_dict_for(sorted_extra_dict, {1:[]}, used_codes, min_freq=1)

        # Augment extra words
        augmented_extra_dict = augment_common_words(sorted_extra_dict, [augmented_phrases, *abbreviated_dicts, *abbreviated_extra_dicts], preemptive=False)

        # Split extra words into parts
        boundaries = [3, max(augmented_extra_dict.keys())]
        length = 0
        extra_tables = []
        for i in range(len(boundaries)):
            upper = boundaries[i]
            name = args.name + extra_suffix + str(i)
            extra_tables.append(name)

            augmented_dict_part = dict()
            while length < upper:
                length += 1
                if length not in augmented_extra_dict:
                    continue
                augmented_dict_part[length] = augmented_extra_dict[length]

            filename = path + name + file_suffix
            with open(filename, 'w', encoding='utf-8') as f:
                # Print extra words from input files
                print(get_header(name), file=f)
                print_word_codes(augmented_dict_part, f)
            print(f"Extra words of part {i} written to {name + file_suffix}")

        # Update input_tables
        input_tables.extend(extra_tables)

    if args.abbreviate and args.input_files and args.extra_table:
        name = args.name + abbrevextra_suffix
        filename = path + name + file_suffix
        with open(filename, 'w', encoding='utf-8') as f:
            # Print abbreviated codes for most frequent words
            print(get_header(name), file=f)
            for abbreviated_dict in abbreviated_extra_dicts:
                print_word_codes(abbreviated_dict, f)
        print(f"Abbreviated extra codes written to {name + file_suffix}")

    if args.chinese_code:
        name = args.name
        filename = path + name + file_suffix
        with open(filename, 'w', encoding='utf-8') as f:
            # Print Chinese character codes
            print(get_header(name, input_tables), file=f)
            print_word_codes(augmented_characters, f, freq_base=10000)
        print(f"Chinese character codes written to {name + file_suffix}")

    if args.test:
        # Run unit tests
        unittest.main(argv=[sys.argv[0]], exit=False)

if __name__ == "__main__":
    main()



