with open('language_encoder.txt', 'r') as f:
    lines = f.readlines()
    #print(lines)

LANGUAGE_NUM = len(lines)-2
print(LANGUAGE_NUM)

for idx, line in enumerate(lines):
    if idx < LANGUAGE_NUM:
        print(f'Line {idx+1}: {line[1:3]}\t Index: {int(line[8])}')
        print(type(str(line[1:3])))