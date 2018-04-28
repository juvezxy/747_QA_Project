# coding:utf-8
from config import *

def is_digit_word(word):
    return re.match(r'\d+', word)

def vars_from_data(data):
    question, answer, question_embedded, answer_embedded, answer_ids, kb_position, kb_facts, kb_facts_embedded, answer_modes, answ4ques_locs, answ4kb_locs = data
    ques_var = Variable(question_embedded)
    answ_var = Variable(answer_embedded)
    answ_id_var = Variable(torch.LongTensor(answer_ids).view(-1, 1))
    kb_position_var = Variable(torch.LongTensor(kb_position).view(-1))
    kb_var = Variable(kb_facts_embedded)
    answer_modes_var = [Variable(torch.LongTensor([answer_mode]).view(-1)) for answer_mode in answer_modes]
    answ4ques_locs_var = [Variable(torch.LongTensor(answ4ques_loc).view(1, 1, -1)) for answ4ques_loc in answ4ques_locs]
    answ4kb_locs_var = [Variable(torch.LongTensor(answ4kb_loc).view(1, 1, -1)) for answ4kb_loc in answ4kb_locs]

    if use_cuda:
        answer_modes_var = [answer_mode_var.cuda() for answer_mode_var in answer_modes_var]
        answ4ques_locs_var = [answ4ques_loc_var.cuda() for answ4ques_loc_var in answ4ques_locs_var]
        answ4kb_locs_var = [answ4kb_loc_var.cuda() for answ4kb_loc_var in answ4kb_locs_var]
        return (ques_var.cuda(), answ_var.cuda(), kb_var.cuda(), kb_position_var.cuda(), answ_id_var.cuda(), answer_modes_var, answ4ques_locs_var, answ4kb_locs_var, kb_facts, question, answer)
    else:
        return (ques_var, answ_var, kb_var, kb_position_var, answ_id_var, answer_modes_var, answ4ques_locs_var, answ4kb_locs_var, kb_facts, question, answer)

def tokenizer(sentence, ent=None):
    tokenized_list = []
    if ent is not None:
        try:
            index = sentence.index(ent)
            preEnt = sentence[:index]
            postEnt = sentence[index+len(ent):]
            tokenized_list = [token for token in preEnt.split(' ') if token != ''] + [ent] + [token for token in postEnt.split(' ') if token != '']
        except:
            tokenized_list = sentence.split(' ')
    else:
        tokenized_list = sentence.split(' ')
    #print (tokenized_list, ent)
    return tokenized_list

class WordIndexer:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "_SOS", 1: "_EOS", 2: "_PAD", 3: "_UNK", 4: "_FIL"}
        self.wordCount = 5

    # index each word in the sentence and return a list of indices
    def addSentence(self, sentence, entity=list(), ent=None):
        indexList = []
        tokenized = tokenizer(sentence, ent)
        for word in tokenized:
            if word not in entity and not is_digit_word(word):
                indexList.append(self.addWord(word))
            else:
                indexList.append(PAD)
        # append EOS at end of each sentence
        indexList.append(EOS)
        tokenized.append("_EOS")
        return tokenized, indexList

    def indexSentence(self, sentence, entity=list(), ent=None):
        indexList = []
        tokenized = tokenizer(sentence, ent)
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
        if count <= 2:
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
        #self.qa_data_path = data_path + "qa_pairs"
        #self.kb_data_path = data_path + "kb_facts"
        self.train_path = data_path + "train"
        self.test_path = data_path + "dev"
        self.is_cqa = is_cqa
        self.min_frq = min_frq
        self.max_vocab_size = max_vocab_size
        self.max_fact_num = 1
        self.max_ques_len = 10
        self.load_data()

    def normalize(self, sent):
        return ''.join(token for token in sent.lower() if token not in string.punctuation)

    def read_QA_pairs(self, qa_data_path):
        qaPairs = []
        with open(qa_data_path, 'r', encoding='utf-8') as inputFile:
            for line in inputFile:
                triple_index = line.index('fact:')
                q_a, triples = line[:triple_index], line[triple_index:]
                answer_index = q_a.index('answer:')
                question, answer = q_a[9:answer_index].strip(), q_a[answer_index+7:].strip()
                question = self.normalize(question)
                answer = self.normalize(answer)
                #self.max_ques_len = max(len(tokenizer(question)), self.max_ques_len)
                
                triples = triples.split('fact:')[1:]
                triple_list = list()
                for triple in triples:
                    parts = triple.split('\t')
                    if len(parts) < 3: # only contains object
                        sub, rel, obj = '_PAD', '_PAD', self.normalize(parts[0].strip())
                    else:
                        sub, rel, obj = self.normalize(parts[0].strip()), self.normalize(parts[1].strip()), self.normalize(parts[2].strip())
                    triple_list.append((sub, rel, obj))
                if len(triple_list) < 1:
                    continue
                if len(tokenizer(question)) > self.max_ques_len:
                    continue

                qaPairs.append((question, answer, triple_list))
        return qaPairs

    def load_data(self):
        self.wordIndexer = WordIndexer()
        self.testing_data = []
        self.training_data = []
        qaPairs = []

        # KB facts
        self.entity_facts = dict()
        entities, relations = set(), set()
    
        with open(self.train_path, 'r', encoding='utf-8') as inputFile:
            for line in inputFile:
                triple_index = line.index('fact:')
                triples = line[triple_index:]
                triples = triples.split('fact:')[1:]
                for triple in triples:
                    parts = triple.split('\t')
                    if len(parts) < 3: # only contains object
                        sub, rel, obj = '_PAD', '_PAD', self.normalize(parts[0].strip())
                    else:
                        sub, rel, obj = self.normalize(parts[0].strip()), self.normalize(parts[1].strip()), self.normalize(parts[2].strip())

                    entities.add(sub)
                    entities.add(obj)

                    relations.add(rel)

                    facts = self.entity_facts.get(sub, set())
                    facts.add((sub, rel, obj))
                    self.entity_facts[sub] = facts
        with open(self.test_path, 'r', encoding='utf-8') as inputFile:
            for line in inputFile:
                triple_index = line.index('fact:')
                triples = line[triple_index:]
                triples = triples.split('fact:')[1:]
                for triple in triples:
                    parts = triple.split('\t')
                    if len(parts) < 3: # only contains object
                        sub, rel, obj = '_PAD', '_PAD', self.normalize(parts[0].strip())
                    else:
                        sub, rel, obj = self.normalize(parts[0].strip()), self.normalize(parts[1].strip()), self.normalize(parts[2].strip())

                    entities.add(sub)
                    entities.add(obj)

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
        print("Max KB fact size: ", self.max_fact_num)

        # QA pairs
        print('Reading from file', self.train_path, '...')
        train_pairs = self.read_QA_pairs(self.train_path)
        print('Reading from file', self.test_path, '...')
        test_pairs = self.read_QA_pairs(self.test_path)        
        self.max_ques_len += 1
        print(len(train_pairs), 'training pairs read.')
        print(len(test_pairs), 'test pairs read.')
        print('Maximum question length: ', self.max_ques_len)
        #shuffle(train_pairs)
        #shuffle(test_pairs)
        qaPairs = train_pairs + test_pairs

        split = len(train_pairs)
        self.gold_answer = list()

        # Counting words
        for i in range(len(qaPairs)):
            question, answer, triple_list = qaPairs[i]
            is_training_data = i < split
            if is_training_data:
                self.wordIndexer.count_sentence(question, self.kb_entities)
        # Indexing words
        matched_pair = 0
        embedder = ElmoEmbedder()
        for i in range(len(qaPairs)):
            question, answer, triple_list = qaPairs[i]
            #print (qaPairs[i])
            is_training_data = i < split
            if is_training_data:
                question, question_ids = self.wordIndexer.addSentence(question, self.kb_entities)
                answer, answer_ids = self.wordIndexer.addSentence(answer, self.kb_entities, triple_list[0][2])
            else:
                question, question_ids = self.wordIndexer.indexSentence(question, self.kb_entities)
                answer, answer_ids = self.wordIndexer.indexSentence(answer, self.kb_entities, triple_list[0][2])
            for pad_to_max in range(self.max_ques_len - len(question_ids)):
                question.append('_FIL')

            kb_facts,kb_facts_ids = [], []
            
            kb_facts = [triple_list[0]]
            '''
            first_entity = triple_list[0][0]
            for triple in self.entity_facts.get(first_entity, list()):
                if triple != triple_list[0]:
                    kb_facts.append(triple)

            if len(kb_facts) > self.max_fact_num:
                shuffle(kb_facts)
                kb_facts = kb_facts[:self.max_fact_num]
            else:
                for pad_index in range(self.max_fact_num - len(kb_facts)):
                    kb_facts.append(("_FIL", "_FIL", "_FIL"))
            '''
            for (sub, rel, obj) in kb_facts:
                kb_facts_ids.append((self.wordIndexer.word2index.get(rel, PAD),self.wordIndexer.word2index.get(obj, PAD)))

            fact_objs = [x[2] for x in kb_facts]

            answer_modes, has_kb_matched = [], False
            answ4ques_locs, answ4kb_locs = [], []
            for word in answer:
                if word in fact_objs: # mode 1: retrieve mode
                    if not has_kb_matched:
                        has_kb_matched = True
                        matched_pair += 1
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
            # Textualized representation
            question_embedded = torch.from_numpy(embedder.embed_sentence(question)).transpose(0,1).contiguous().view(len(question),1,-1)
            answer_embedded = [torch.from_numpy(embedder.embed_sentence([word]))[:,0][0].view(1,1,-1) for word in answer]
            answer_embedded = torch.cat(answer_embedded, 0)
            kb_fact_embedded = torch.from_numpy(embedder.embed_sentence([kb_facts[0][-1]]))[:,0][0].view(1,1,-1)
            kb_position = -1
            try:
                kb_position = answer.index(kb_facts[0][-1])
            except:

                for kb_try in kb_facts[0][-1].split(" "):
                    try:
                        kb_position = answer.index(kb_try)
                    except:
                        continue
            if kb_position == -1:
                kb_position = 5

            if i % 1000 == 0:
                print(i, "pair")

            if (True):
                if is_training_data:
                    self.training_data.append((question, answer, question_embedded, answer_embedded, answer_ids, kb_position, kb_facts, kb_fact_embedded,
                                               answer_modes, answ4ques_locs, answ4kb_locs))
                else:
                    self.testing_data.append((question, answer, question_embedded, answer_embedded, answer_ids, kb_position, kb_facts, kb_fact_embedded,
                                               answer_modes, answ4ques_locs, answ4kb_locs))
                    self.gold_answer.append(triple_list)

            if i % 10000 == 0 and i / 10000 > 0:
                print("Saving to preprocessed file ...")
                with open(preprocessed_data_path+"training"+str(i / 10000), 'wb') as preprocessed:
                    pickle.dump(self.training_data, preprocessed)
                with open(preprocessed_data_path+"testing"+str(i / 10000), 'wb') as preprocessed:
                    pickle.dump(self.testing_data, preprocessed)
                self.training_data = []
                self.testing_data = []

        print("Saving to preprocessed file ...")
        with open(preprocessed_data_path + "training6", 'wb') as preprocessed:
            pickle.dump(self.training_data, preprocessed)
        with open(preprocessed_data_path + "testing6", 'wb') as preprocessed:
            pickle.dump(self.testing_data, preprocessed)
        self.training_data = []
        self.testing_data = []

        print('Processing done.', len(self.training_data), 'training pairs,', len(self.testing_data), 'test pairs.')
        print('Total vocab size: ', self.wordIndexer.wordCount)







