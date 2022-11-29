import json


def main():
    tokens = {}
    print("building dictionary...")

    id = 1 # reserve id 0 to mark the end of a string

    with open('./completions' + ".dict", "r", encoding='utf-8') as f:
        for line in f:
            t = line.rstrip('\n')
            tokens[t] = id
            id += 1
    lines = 0
    print("mapping dataset...")

    output_file = open('completions' + ".mapped", 'w', encoding='utf-8')
    stats_file = open('completions' + ".mapped.stats", 'w', encoding='utf-8')

    max_string_len = 0
    with open('completions_collect_word_level.txt', 'r') as f:
        for line in f:
            x = line.rstrip('\n').split()
            string_len = 0
            mapped = [x[0]]
            for i in range(1, len(x)): # x[0] stores the docID
                t = x[i]
                try:
                    id = tokens[t]
                    mapped.append(id)
                    string_len += len(t)
                except KeyError:
                    print("'" + t + "' not found in dictionary")
                    print(line)
                    exit()
            
            if string_len > max_string_len:
                max_string_len = string_len
            mapped.append("0") # terminator
            s = [str(i) for i in mapped]
            output_file.write(" ".join(s) + "\n")

            lines += 1
            if lines % 5000 == 0:
                print("processed " + str(lines) + " lines")

    output_file.close()

    stats_file.write(str(len(tokens)) + "\n")
    stats_file.write(str(max_string_len) + "\n")
    stats_file.close()

    with open('completions.dict.json', 'w', encoding='utf-8') as f:
        json.dump(tokens, f, ensure_ascii=False)
if __name__ == '__main__':
    main()