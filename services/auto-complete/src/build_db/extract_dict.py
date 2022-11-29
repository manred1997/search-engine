


def main():

    tokens = set()
    lines = 0

    print("parsing input file...")
    with open('./completions_collect_word_level.txt', 'r', encoding='utf-8') as f:
        for line in f:
            x = line.rstrip('\n').split()
            for i in range(1, len(x)):
                tokens.add(x[i])
            lines += 1
        if lines % 5000 == 0:
            print("processed " + str(lines) + " lines")
    
    print("processed " + str(lines) + " lines")
    print("dictionary has " + str(len(tokens)) + " keys")

    dict_file = open('completions' + ".dict", 'w')
    for key in sorted(tokens):
        dict_file.write(key + "\n")
        # dict_file.write(key.replace('_', ' ') + "\n")
    dict_file.close()

if __name__ == '__main__':
    main()