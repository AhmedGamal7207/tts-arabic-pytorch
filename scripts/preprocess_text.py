# This file needs to be run in the main folder
# %%
import text
from utils import read_lines_from_file


def write_lines_to_file(path, lines, mode='w', encoding='utf-8'):
    with open(path, mode, encoding=encoding) as f:
        for i, line in enumerate(lines):
            if i == len(lines)-1:
                f.write(line)
                break
            f.write(line + '\n')

# %%


lines = read_lines_from_file('/content/tts-arabic-pytorch/Mini Arabic Youtube Videos/train_buckwalter.txt')
#lines = read_lines_from_file('/content/tts-arabic-pytorch/Mini Arabic Youtube Videos/test_buckwalter.txt')

new_lines_arabic = []
new_lines_phonetic = []
new_lines_buckw = []

for line in lines:
    wav_name, utterance = line.split('" "')
    wav_name, utterance = wav_name[1:], utterance[:-1]
    utterance = utterance.replace("a~", "~a") \
                         .replace("i~", "~i") \
                         .replace("u~", "~u") \
                         .replace(" - ", " ")

    utterance_arab = text.buckwalter_to_arabic(utterance)
    utterance_phon = text.buckwalter_to_phonemes(utterance)

    line_new_ara = f'"{wav_name}" "{utterance_arab}"'
    new_lines_arabic.append(line_new_ara)

    line_new_pho = f'"{wav_name}" "{utterance_phon}"'
    new_lines_phonetic.append(line_new_pho)

    line_new_buckw = f'"{wav_name}" "{utterance}"'
    new_lines_buckw.append(line_new_buckw)


# %% train

write_lines_to_file('/content/tts-arabic-pytorch/Mini Arabic Youtube Videos/final_text_files/train_arab.txt', new_lines_arabic)
write_lines_to_file('/content/tts-arabic-pytorch/Mini Arabic Youtube Videos/final_text_files/train_phon.txt', new_lines_phonetic)
write_lines_to_file('/content/tts-arabic-pytorch/Mini Arabic Youtube Videos/final_text_files/train_buckw.txt', new_lines_buckw)
'''
# %% test

write_lines_to_file('/content/tts-arabic-pytorch/Mini Arabic Youtube Videos/final_text_files/test_arab.txt', new_lines_arabic)
write_lines_to_file('/content/tts-arabic-pytorch/Mini Arabic Youtube Videos/final_text_files/test_phon.txt', new_lines_phonetic)
write_lines_to_file('/content/tts-arabic-pytorch/Mini Arabic Youtube Videos/final_text_files/test_buckw.txt', new_lines_buckw)
'''