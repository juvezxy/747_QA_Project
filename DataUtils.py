# coding:utf-8
from config import *

def is_digit_word(word):
    return re.match(r'\d+', word)

def vars_from_data(data):
    question, answer, question_ids, answer_ids, kb_facts, kb_facts_ids, answer_modes, answ4ques_locs, answ4kb_locs = data
    ques_var = Variable(torch.LongTensor(question_ids).view(-1, 1))
    answ_var = Variable(torch.LongTensor(answer_ids).view(-1, 1))
    kb_var = [Variable(torch.LongTensor(kb_fact_id).view(-1, 1)) for kb_fact_id in kb_facts_ids]
    answer_modes_var = [Variable(torch.LongTensor([answer_mode]).view(-1)) for answer_mode in answer_modes]
    answ4ques_locs_var = [Variable(torch.LongTensor(answ4ques_loc).view(1, 1, -1)) for answ4ques_loc in answ4ques_locs]
    answ4kb_locs_var = [Variable(torch.LongTensor(answ4kb_loc).view(1, 1, -1)) for answ4kb_loc in answ4kb_locs]

    if use_cuda:
        kb_var = [kb_fact_var.cuda() for kb_fact_var in kb_var]
        answer_modes_var = [answer_mode_var.cuda() for answer_mode_var in answer_modes_var]
        answ4ques_locs_var = [answ4ques_loc_var.cuda() for answ4ques_loc_var in answ4ques_locs_var]
        answ4kb_locs_var = [answ4kb_loc_var.cuda() for answ4kb_loc_var in answ4kb_locs_var]
        return (ques_var.cuda(), answ_var.cuda(), kb_var, answer_modes_var, answ4ques_locs_var, answ4kb_locs_var, kb_facts, question, answer)
    else:
        return (ques_var, answ_var, kb_var, answer_modes_var, answ4ques_locs_var, answ4kb_locs_var, kb_facts, question, answer)

def tokenizer(sentence):
    tokenized_list = []
    entMatch = entPattern.search(sentence)
    if (entMatch):
        preEnt = sentence[:entMatch.start()]
        ent = entMatch.group()
        postEnt = sentence[entMatch.end():]
        tokenized_list = list(jieba.cut(preEnt, cut_all=False)) + [ent] + list(jieba.cut(postEnt, cut_all=False))
    else:
        tokenized_list = [token for token in jieba.cut(sentence, cut_all=False) if token not in string.whitespace and token not in string.punctuation]
    return tokenized_list

class WordIndexer:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "_SOS", 1: "_EOS", 2: "_PAD", 3: "_UNK", 4: "_FIL"}
        self.wordCount = 5

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
        tokenized.append("_EOS")
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
        tokenized.append("_EOS")
        return tokenized, indexList

    def count_add_word(self, word):
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

    # index the word and return its correponding index
    def addWord(self, word):
        count = self.word2count.get(word, 0)
        # only keep words with freq > 3 
        if count <= 30:
            self.word2index[word] = UNK
            return UNK

        if word not in self.word2index:
            index = self.wordCount
            self.word2index[word] = self.wordCount
            self.index2word[self.wordCount] = word
            self.wordCount += 1
        else:
            index = self.word2index[word]
        return index

    def count_sentence(self, sentence, entity=list()):
        tokenized = tokenizer(sentence)
        for word in tokenized:
            if word not in entity and not is_digit_word(word):
                self.count_word(word)

    def count_word(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1


class DataLoader(object):
    def __init__(self, data_path, is_cqa, min_frq=0, max_vocab_size=0):
        self.qa_data_path = data_path + "qa_pairs"
        self.kb_data_path = data_path + "kb_facts"
        self.is_cqa = is_cqa
        self.min_frq = min_frq
        self.max_vocab_size = max_vocab_size
        self.max_fact_num = 0
        self.max_ques_len = 10
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
        if self.is_cqa:
            with open(self.qa_data_path, 'r', encoding='utf-8') as inputFile:
                for line in inputFile:
                    triple_index = line.index('triple')
                    triples = line[triple_index:]
                    triples = triples.split('triple')[1:]
                    for triple in triples:
                        triple = triple[2:]
                        parts = triple.split()
                        if len(parts) < 3:
                            continue
                        sub, rel, obj = parts[0].strip(), parts[1].strip(), parts[2].strip()
                        # TODO: Improve the KB embedding/how to interpret KB
                        entities.add(sub)
                        #if not is_digit_word(obj):
                            #self.wordIndexer.count_add_word(obj)
                        relations.add(rel)

                        facts = self.entity_facts.get(sub, set())
                        facts.add((sub, rel, obj))
                        self.entity_facts[sub] = facts
        else:
            with open(self.kb_data_path, 'r', encoding='utf-8') as inputFile:
                for line in inputFile:
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    sub, rel, obj = [w.strip() for w in parts]
                    # TODO: Improve the KB embedding/how to interpret KB
                    entities.add(sub)
                    if not is_digit_word(obj):
                        self.wordIndexer.count_add_word(obj)
                    relations.add(rel)

                    facts = self.entity_facts.get(sub, set())
                    facts.add((sub, rel, obj))
                    self.entity_facts[sub] = facts

        self.kb_relations = relations
        self.kb_entities = entities
        for rel in self.kb_relations:
            self.wordIndexer.count_add_word(rel)
        for sub in self.entity_facts.keys():
            self.entity_facts[sub] = list(self.entity_facts[sub])
        print("KB entity size: ", len(self.entity_facts))
        print("KB fact size: ", sum([len(x) for x in self.entity_facts.values()]))
        self.max_fact_num = 4
        print("Max KB fact size: ", self.max_fact_num)

        # QA pairs
        print('Reading from file', self.qa_data_path, '...')
        with open(self.qa_data_path, 'r', encoding='utf-8') as inputFile:
            for line in inputFile:
                if self.is_cqa:
                    triple_index = line.index('triple')
                    q_a, triples = line[:triple_index], line[triple_index:]
                    answer_index = q_a.index('answer:')
                    question, answer = q_a[9:answer_index].strip(), q_a[answer_index+7:].strip()
                    #self.max_ques_len = max(len(tokenizer(question)), self.max_ques_len)
                    if len(tokenizer(question)) > self.max_ques_len:
                        continue
                    triples = triples.split('triple')[1:]
                    triple_list = list()
                    for triple in triples:
                        triple = triple[2:]
                        parts = triple.split()
                        if len(parts) < 3:
                            continue
                        sub, rel, obj = parts[0].strip(), parts[1].strip(), parts[2].strip()
                        triple_list.append((sub, rel, obj))
                    qaPairs.append((question, answer, triple_list))
                else:
                    question, answer = line.split()
                    self.max_ques_len = max(len(tokenizer(question)), self.max_ques_len)
                    qaPairs.append((question, answer))
        self.max_ques_len = 11
        print(len(qaPairs), 'pairs read.')
        print('Maximum question length: ', self.max_ques_len)
        shuffle(qaPairs)

        split = int(0.9 * len(qaPairs))
        self.gold_answer = list()

        # Counting words
        for i in range(len(qaPairs)):
            if self.is_cqa:
                question, answer, triple_list = qaPairs[i]
            else:
                question, answer = qaPairs[i]
            is_training_data = i < split
            if is_training_data:
                self.wordIndexer.count_sentence(question, self.kb_entities)
        # Indexing words
        for i in range(len(qaPairs)):
            if self.is_cqa:
                question, answer, triple_list = qaPairs[i]
            else:
                question, answer = qaPairs[i]
            is_training_data = i < split
            if is_training_data:
                question, question_ids = self.wordIndexer.addSentence(question, self.kb_entities)
                answer, answer_ids = self.wordIndexer.addSentence(answer, self.kb_entities)
            else:
                question, question_ids = self.wordIndexer.indexSentence(question, self.kb_entities)
                answer, answer_ids = self.wordIndexer.indexSentence(answer, self.kb_entities)
            for pad_to_max in range(self.max_ques_len - len(question_ids)):
                question_ids.append(FIL)

            kb_facts,kb_facts_ids = [], []
            for word in question:
                kb_facts += self.entity_facts.get(word, list())
            if len(kb_facts) > self.max_fact_num:
                shuffle(kb_facts)
                kb_facts = kb_facts[:self.max_fact_num]
            else:
                for pad_index in range(self.max_fact_num - len(kb_facts)):
                    kb_facts.append(("_FIL", "_FIL", "_FIL"))
            for (sub, rel, obj) in kb_facts:
                kb_facts_ids.append((self.wordIndexer.word2index.get(rel, PAD),self.wordIndexer.word2index.get(obj, PAD)))
            fact_objs = [x[2] for x in kb_facts]

            answer_modes = []
            answ4ques_locs, answ4kb_locs = [], []
            for word in answer:
                if word in fact_objs: # mode 1: retrieve mode
                    answer_modes.append(1)
                    kb_locs = list()
                    for obj in fact_objs:
                        if obj == word:
                            kb_locs.append(1)
                        else:
                            kb_locs.append(0)
                    answ4ques_locs.append([0]*self.max_ques_len)
                    answ4kb_locs.append(kb_locs)
                elif word in question and word in self.kb_entities: # mode 2: copy mode
                    answer_modes.append(2)
                    ques_locs = list()
                    for qword in question:
                        if qword == word:
                            ques_locs.append(1)
                        else:
                            ques_locs.append(0)
                    for pad_to_max in range(self.max_ques_len - len(question)):
                        ques_locs.append(0)
                    answ4ques_locs.append(ques_locs)
                    answ4kb_locs.append([0]*self.max_fact_num)
                else: # mode 0: predict mode
                    answer_modes.append(0)
                    answ4ques_locs.append([0]*self.max_ques_len)
                    answ4kb_locs.append([0]*self.max_fact_num)
            if is_training_data:
                self.training_data.append((question, answer, question_ids, answer_ids, kb_facts, kb_facts_ids,
                                           answer_modes, answ4ques_locs, answ4kb_locs))
            else:
                self.testing_data.append((question, answer, question_ids, answer_ids, kb_facts, kb_facts_ids,
                                           answer_modes, answ4ques_locs, answ4kb_locs))
                if self.is_cqa:
                    self.gold_answer.append(triple_list)

        print('Processing done.', len(self.training_data), 'training pairs,', len(self.testing_data), 'test pairs.')
        print('Total vocab size: ', self.wordIndexer.wordCount)







