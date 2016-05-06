import sys, string, random, numpy
from nltk.corpus import reuters
from llda import LLDA
from optparse import OptionParser


parser = OptionParser()
parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.001)
parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.001)
parser.add_option("-k", dest="K", type="int", help="number of topics", default=50)
parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
parser.add_option("-s", dest="seed", type="int", help="random seed", default=None)
parser.add_option("-n", dest="samplesize", type="int", help="dataset sample size", default=100)
(options, args) = parser.parse_args()
random.seed(options.seed)
numpy.random.seed(options.seed)

def word_distribution(set_tags, posts, tags):
    tag_dict = {}
    corpus = []
    labels = []
    labelset = []
    for post in posts:
        row_p = []
        for p in post:
            # p_u = unicode(p, "utf-8")
            row_p.append(p)
        corpus.append(row_p)
    for tag in tags:
        row_t = []
        for t in tag:
            # t_u = unicode(t, "utf-8")
            row_t.append(t)
        labels.append(row_t)
    for st in set_tags:
        # st_u = unicode(st, "utf-8")
        labelset.append(st)
    llda = LLDA(options.K, options.alpha, options.beta)
    llda.set_corpus(labelset, corpus, labels)

    for i in range(options.iteration):
        sys.stderr.write("-- %d : %.4f\n" % (i, llda.perplexity()))
        llda.inference()
    print "perplexity : %.4f" % llda.perplexity()

    phi = llda.phi()
    for k, label in enumerate(set_tags):
        print "\n-- label %d : %s" % (k, label)
        vocab_dict = {}
        for w in numpy.argsort(-phi[k])[:20]:
            print "%s: %.4f" % (llda.vocas[w], phi[k,w])

            vocab_dict[llda.vocas[w]] = phi[k,w]

        tag_dict[label] = vocab_dict
    return tag_dict