BOS = 'bos'
EOS = 'eos'

with open('../data/glove.6B.300d.txt', 'r',encoding='utf-8') as f:
    flag = False
    counter=1
    for line in f:
        if 'bos' in line or '<s>' in line:
            flag = True
            print("BOS:{} {}".format(line.split()[0],counter))
        if 'eos' in line or '</s>' in line:
            flag = True
            print("BOS:{} {}".format(line.split()[0],counter))
        counter+=1
        if flag:
            break
