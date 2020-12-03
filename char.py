with open("table_lmdb_dataset/dict.txt", "r") as f:
    t = []
    for i in f.readlines():
        t.append(i.strip('\n'))
    character = set(t)
    character.update('\u2028')

with open("char.txt", "w") as f:
    f.write(''.join(character))