# coding:utf-8
from config import *

def is_digit_word(word):
    return re.match(r'\d+', word)

def vars_from_data(data):
    question, answer, question_ids, answer_ids, kb_facts, kb_facts_ids, answer_modes, answ4ques_locs, answ4kb_locs = data
    ques_var = Variable(torch.LongTensor(question_ids).view(-1, 1))
    answ_var = Variable(torch.LongTensor(answer_ids).view(-1, 1))
    kb_var = [Variable(torch.LongTensor(kb_fact_id).view(-1, 1)) for kb_fact_id in kb_facts_ids]
    if use_cuda:
        kb_var = [kb_fact_var.cuda() for kb_fact_var in kb_var]
        return (ques_var.cuda(), answ_var.cuda(), kb_var)
    else:
        return (ques_var, answ_var, kb_var)

def tokenizer(sentence):
    tokenized_list = []
    entMatch = entPattern.search(sentence)
    if (entMatch):
        preEnt = sentence[:entMatch.start()]
        ent = entMatch.group()
        postEnt = sentence[entMatch.end():]
        tokenized_list = list(jieba.cut(preEnt, cut_all=False)) + [ent] + list(jieba.cut(postEnt, cut_all=False))
    else:
        tokenized_list = list(jieba.cut(sentence, cut_all=False))
    return tokenized_list

class WordIndexer:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "_SOS", 1: "_EOS", 2: "_PAD", 3: "_UNK"}
        self.wordCount = 4

    # index each word in the sentence and return a list of indices
    def addSentence(self, sentence, entity=list()):
        indexList = []
        tokenized = tokenizer(sentence)
        for word in tokenized:
            if word not in entity and not is_digit_word(word):
                indexList.append(self.addWord(word))
            else:
                indexList.append(PAD)
        # append EOS at end of each sentence
        indexList.append(EOS)
        return tokenized, indexList

    def indexSentence(self, sentence, entity=list()):
        indexList = []
        tokenized = tokenizer(sentence)
        for word in tokenized:
            if word not in entity and not is_digit_word(word):
                indexList.append(self.word2index.get(word, UNK))
            else:
                indexList.append(PAD)
        # append EOS at end of each sentence
        indexList.append(EOS)
        return tokenized, indexList

    # index the word and return its correponding index
    def addWord(self, word):
        if word not in self.word2index:
            index = self.wordCount
            self.word2index[word] = self.wordCount
            self.word2count[word] = 1
            self.index2word[self.wordCount] = word
            self.wordCount += 1
        else:
            self.word2count[word] += 1
            index = self.word2index[word]
        return index

class DataLoader(object):
    def __init__(self, data_path, min_frq=0, max_vocab_size=0):
        self.qa_data_path = data_path + "qa_pairs"
        self.kb_data_path = data_path + "kb_facts"
        self.min_frq = min_frq
        self.max_vocab_size = max_vocab_size
        self.max_fact_num = 4
        self.load_data()


    # jieba.load_userdict(config.userDictPath)
    def load_data(self):
        self.wordIndexer = WordIndexer()
        self.testing_data = []
        self.training_data = []
        qaPairs = []

        # KB facts
        print('Reading from file', self.kb_data_path, '...')
        self.entity_facts = dict()
        entities, relations = set(), set()
        with open(self.kb_data_path, 'r', encoding='utf-8') as inputFile:
            for line in inputFile:
                parts = line.split()
                if len(parts) < 3:
                    continue
                sub, rel, obj = [w.strip() for w in parts]
                # TODO: Improve the KB embedding/how to interpret KB
                entities.add(sub)
                if not is_digit_word(obj):
                    self.wordIndexer.addWord(obj)
                relations.add(rel)

                facts = self.entity_facts.get(sub, list())
                facts.append((sub, rel, obj))
                self.entity_facts[sub] = facts
        self.kb_relations = relations
        self.kb_entities = entities
        for rel in self.kb_relations:
            self.wordIndexer.addWord(rel)
        for sub in self.entity_facts.keys():
            self.entity_facts[sub] = sorted(self.entity_facts[sub], key=lambda x: x[0])
        print("KB entity size: ", len(self.entity_facts))
        print("KB fact size: ", sum([len(x) for x in self.entity_facts.values()]))

        # QA pairs
        print('Reading from file', self.qa_data_path, '...')
        with open(self.qa_data_path, 'r', encoding='utf-8') as inputFile:
            for line in inputFile:
                question, answer = line.split()
                qaPairs.append((question, answer))
        print(len(qaPairs), 'pairs read.')
        shuffle(qaPairs)

        split = int(0.9 * len(qaPairs))
        # Training data
        for i in range(len(qaPairs)):
            question, answer = qaPairs[i]
            is_training_data = i < split
            if is_training_data:
                question, question_ids = self.wordIndexer.addSentence(question, self.kb_entities)
                answer, answer_ids = self.wordIndexer.addSentence(answer, self.kb_entities)
            else:
                question, question_ids = self.wordIndexer.indexSentence(question, self.kb_entities)
                answer, answer_ids = self.wordIndexer.indexSentence(answer, self.kb_entities)

            kb_facts,kb_facts_ids = [], []
            for word in question:
                kb_facts += self.entity_facts.get(word, list())
            if len(kb_facts) > self.max_fact_num:
                shuffle(kb_facts)
                kb_facts = kb_facts[:self.max_fact_num]
            else:
                for pad_index in range(self.max_fact_num - len(kb_facts)):
                    kb_facts.append(("_PAD", "_PAD", "_PAD"))
            for (sub, rel, obj) in kb_facts:
                kb_facts_ids.append((self.wordIndexer.word2index.get(rel, PAD),self.wordIndexer.word2index.get(obj, PAD)))
            fact_objs = [x[2] for x in kb_facts]

            answer_modes = []
            answ4ques_locs, answ4kb_locs = [], []
            for word in answer:
                # TODO: add copy mode
                if word in fact_objs: # mode 1: retrieve mode
                    answer_modes.append(1)
                    kb_locs = list()
                    for obj in fact_objs:
                        if obj == word:
                            kb_locs.append(1)
                        else:
                            kb_locs.append(0)
                    answ4ques_locs.append(list())
                    answ4kb_locs.append(kb_locs)
                else if word in question: # mode 2: copy mode
                    answer_modes.append(2)
                    ques_locs = list()
                    for qword in question:
                        if qword == word:
                            ques_locs.append(1)
                        else:
                            ques_locs.append(0)
                    answ4ques_locs.append(ques_locs)
                    answ4kb_locs.append(list())
                else: # mode 0: predict mode
                    answer_modes.append(0)
                    answ4ques_locs.append(list())
                    answ4kb_locs.append(list())
            if is_training_data:
                self.training_data.append((question, answer, question_ids, answer_ids, kb_facts, kb_facts_ids,
                                           answer_modes, answ4ques_locs, answ4kb_locs))
            else:
                self.testing_data.append((question, answer, question_ids, answer_ids, kb_facts, kb_facts_ids,
                                           answer_modes, answ4ques_locs, answ4kb_locs))

        print('Processing done.', len(self.training_data), 'training pairs,', len(self.testing_data), 'test pairs.')
        print('Total vocab size: ', self.wordIndexer.wordCount)







