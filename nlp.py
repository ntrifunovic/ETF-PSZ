import sys

import jsonrpclib
from simplejson import loads
import csv
import arff

from nltk.corpus import wordnet as wn

import pprint

pp = pprint.PrettyPrinter(indent=4)

server = jsonrpclib.Server("http://localhost:3456")

reader = csv.reader(open(sys.argv[1]), dialect="excel-tab")
next(reader) # skip header

posTypes = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS',
            'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS',
            'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO',
            'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
            'WP$', 'WRB']

depTypes = ['auxpass', 'cop', 'conj', 'cc', 'nsubj', 'csubj', 'dobj',
            'iobj', 'pobj', 'attr', 'ccomp', 'xcomp', 'mark', 'rel',
            'acomp', 'agent', 'ref', 'expl', 'advcl', 'purpcl',
            'tmod', 'rcmod', 'amod', 'infmod', 'partmod', 'num',
            'number', 'appos', 'advmod', 'neg', 'poss', 'possessive',
            'prt', 'det', 'prep', 'xsubj']

words = lambda nlp: nlp['sentences'][0]['words']
dependencies = lambda nlp: nlp['sentences'][0]['dependencies']

word = lambda word: word[0]
pos = lambda word: word[1]['PartOfSpeech']
dep = lambda dependency: dependency[0]

filter_by_type = lambda type, filter1, filter2: lambda nlp: [item for item in filter1(nlp) if filter2(item)==type];

calc = lambda _filter: abs(len(_filter(nlp1))-len(_filter(nlp2)))

out = {
    'relation': 'Short Sentence Similarity',
    'attributes': [
        ('similar', ['yes', 'no']),
        ('diff_All', 'NUMERIC'),
        ('overallLexsim', 'NUMERIC')],
    'data': []
}
    
out['attributes'].extend(map(lambda type: ("diff_Tag_"+type, 'NUMERIC'), posTypes))
out['attributes'].extend(map(lambda type: ("diff_Dep_"+type, 'NUMERIC'), depTypes))
out['attributes'].extend(map(lambda type: ("lexSim_Tag_"+type, 'NUMERIC'), posTypes))
out['attributes'].extend(map(lambda type: ("semSim_Tag_"+type, 'NUMERIC'), posTypes))
out['attributes'].extend(map(lambda type: ("lexSim_Dep_"+type, 'NUMERIC'), depTypes))
out['attributes'].extend(map(lambda type: ("semSim_Dep_"+type, 'NUMERIC'), depTypes))
out['attributes'].extend(map(lambda type: ("same_Tag_"+type, 'NUMERIC'), posTypes))

def lcsubstring_length(a, b):
    table = {}
    l = 0
    for i, ca in enumerate(a, 1):
        for j, cb in enumerate(b, 1):
            if ca == cb:
                table[i, j] = table.get((i - 1, j - 1), 0) + 1
                if table[i, j] > l:
                    l = table[i, j]
    return l

lex_sim = NMCLCSn = lambda a, b: \
  float(lcsubstring_length(a,b)**2)/(len(a)*len(b)) if len(a)*len(b) != 0 else 0
def sem_sim(a, b):
    x = wn.synsets(a)
    y = wn.synsets(b)
    if len(x) == 0 or  len(y) == 0:
         return lex_sim(a,b)
    res = x[0].path_similarity(y[0])
#    print res, a, b
    return  res if res is not None else 0

dot = lambda v1, v2: sum(p*q for p,q in zip(v1, v2))
union = lambda a, b: list(set(a) | set(b))
intersection = lambda a, b: list(set(a) & set(b))

sim_max = lambda word, vector, sim: max(map(lambda w: sim(w, word), vector))
calc_v = lambda st1, st, sim: map(lambda word: sim_max(word, st, sim), st1) 

def s_sim(st1, st2, sim):
    if len(st1) == 0 or len(st2) == 0:
        return 0

    st1 = map(word, st1)
    st2 = map(word, st2)
    if len(st1) == 0 or len(st2) == 0:
        return 0
    st = union(st1, st2)
    v1 = calc_v(st1, st, sim)
    v2 = calc_v(st2, st, sim)
    return float(dot(v1, v2))/(len(v1)*len(v2))

sim_dep = lambda dep1, dep2, sim: sim(dep1[1], dep1[2])*(2**(sim(dep2[1],dep2[2])-1))

def d_sim(st1, st2, sim):
    if len(st1) == 0 or len(st2) == 0:
        return 0

    m = [[0] * (len(st2) + 1) for _ in xrange(len(st1) + 1)]
    for i, rel1 in enumerate(st1, 1):
        for j, rel2 in enumerate(st2, 1):
            m[i][j] = sim_dep(rel1, rel2, sim)

    sum = 0

    while True:
        maxi = maxj = 0
        for i, rel1 in enumerate(st1, 1):
            for j, rel2 in enumerate(st2, 1):
                if(m[i][j] > m[maxi][maxj]):
                    maxi = i
                    maxj = j
        if m[maxi][maxj] == 0:
            break
        sum += m[maxi][maxj]

        for i, rel1 in enumerate(st1, 1):
            m[i][maxj] = 0
        for j, rel2 in enumerate(st2, 1):
            m[maxi][j] = 0

    return float(sum*(len(st1)+len(st2)))/(2*len(st1)*len(st2))


calc_sim = lambda sim, _sim, _filter: sim(_filter(nlp1),
                                          _filter(nlp2), _sim)

i = 0

for line in  reader:
    if len(line) < 5:
        print "ERROR:", line
        continue
    nlp1 = loads(server.parse(line[3]))
    nlp2 = loads(server.parse(line[4]))
    out_line = ['yes' if line[0]=='1' else 'no', calc(words), lex_sim(nlp1,nlp2)]
    #pprint.pprint(nlp1)

    for type in posTypes:
        out_line.append(calc(filter_by_type(type, words, pos)))

    for type in depTypes:
        out_line.append(calc(filter_by_type(type, dependencies, dep)))

    for type in posTypes:
        out_line.append(calc_sim(s_sim, lex_sim, filter_by_type(type, words,  pos)))

    for type in posTypes:
        out_line.append(calc_sim(s_sim, sem_sim, filter_by_type(type, words, pos)))

    for type in depTypes:
        out_line.append(calc_sim(d_sim, lex_sim, filter_by_type(type, dependencies, dep)))

    for type in depTypes:
        out_line.append(calc_sim(d_sim, sem_sim, filter_by_type(type, dependencies, dep)))

    for type in posTypes:
        words1 = map(word,filter_by_type(type, words, pos)(nlp1))
        words2 = map(word,filter_by_type(type, words, pos)(nlp2))
        out_line.append(float(len(intersection(words1,
        words2)))/len(union(words1, words2)) if len(union(words1,
        words2)) > 0 else 0)

    #print out_line
    out['data'].append(out_line)

    i = i+1
    #if i == 100:
    #    break

arff.dump(out, open(sys.argv[2] if len(sys.argv > 2) else 'out.arff', 'w'))
