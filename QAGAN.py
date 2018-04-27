# coding:utf-8

from Encoder import *
from Decoder import *
from DataUtilsNew import *
from config import *


class QAGAN(object):
    def __init__(self, model_params):
        self.word_indexer = model_params["word_indexer"]
        self.embedding_size = model_params["embedding_size"]
        self.state_size = model_params["state_size"]
        self.mode_size = model_params["mode_size"]
        self.ques_attention_size = model_params["ques_attention_size"]
        self.kb_attention_size = model_params["kb_attention_size"]
        self.max_fact_num = model_params["max_fact_num"]
        self.max_ques_len = model_params["max_ques_len"]

        self.learning_rate = model_params["learning_rate"]
        self.mode_loss_rate = model_params["mode_loss_rate"]
        self.L2_factor = model_params["L2_factor"]
        self.batch_size = model_params["batch_size"]
        self.epoch_size = model_params["epoch_size"]
        self.instance_weight = 1.0 / self.batch_size
        self.MAX_LENGTH = model_params["MAX_LENGTH"]
        self.has_trained = False

        ################ Initialize graph components ########################
        self.encoder = Encoder(self.word_indexer.wordCount, self.state_size, self.embedding_size)
        self.decoder = Decoder(output_size=self.word_indexer.wordCount, state_size=self.state_size,
                               embedding_size=self.embedding_size, mode_size=self.mode_size,
                               kb_attention_size=self.kb_attention_size, max_fact_num=self.max_fact_num,
                               ques_attention_size=self.ques_attention_size, max_ques_len=self.max_ques_len)
        if use_cuda:
            self.encoder.cuda()
            self.decoder.cuda()

        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                    lr=self.learning_rate, weight_decay=self.L2_factor)
        #####################################################################

    def fit(self, training_data):
        if self.has_trained:
            print('Warning! Trying to fit a trained model.')

        print('Start training ...')
        startTime = time.time()
        lossTotal = 0.0
        XEnLoss = nn.CrossEntropyLoss()
        for epoch in range(self.epoch_size):
            self.optimizer.zero_grad()
            shuffle(training_data)
            for iter in range(len(training_data)):
                ques_var, answ_var, kb_var_list, answer_modes_var_list, answ4ques_locs_var_list, answ4kb_locs_var_list, kb_facts, ques, answ = vars_from_data(
                    training_data[iter])
                answ_length = answ_var.size()[0]

                #################### Process KB facts ###############################
                kb_facts_embedded = []
                for rel_obj in kb_var_list:
                    rel_embedded = self.embedding(rel_obj[0]).view(1, 1, -1)
                    obj_embedded = self.embedding(rel_obj[1]).view(1, 1, -1)
                    kb_facts_embedded.append(torch.cat((rel_embedded, obj_embedded), 2))
                avg_kb_facts_embedded = kb_facts_embedded[0]
                #####################################################################


                ######################### Encoding ##################################
                self.encoder.hidden = self.encoder.init_hidden()
                encoder_outputs = self.encoder(ques_var)
                question_embedded = self.encoder.hidden[0].view(1, 1, -1)
                cell_state = self.encoder.hidden[1].view(1, 1, -1)
                #####################################################################


                ######################### Decoding ###################################
                decoder_hidden = (question_embedded, cell_state)
                decoder_input = Variable(torch.LongTensor([[SOS]]))
                hist_ques = Variable(torch.zeros(1, 1, self.max_ques_len))
                hist_kb = Variable(torch.zeros(1, 1, self.max_fact_num))
                if use_cuda:
                    decoder_input = decoder_input.cuda()
                    hist_kb = hist_kb.cuda()
                    hist_ques = hist_ques.cuda()

                loss = 0.0
                for i in range(answ_length):
                    answer_mode = answer_modes_var_list[i]
                    word_embedded = self.embedding(decoder_input).view(1, 1, -1)
                    weighted_question_encoding = Variable(torch.zeros(1, 1, 2 * self.state_size))
                    weighted_kb_facts_encoding = Variable(torch.zeros(1, 1, 2 * self.embedding_size))
                    if use_cuda:
                        weighted_question_encoding = weighted_question_encoding.cuda()
                        weighted_kb_facts_encoding = weighted_kb_facts_encoding.cuda()

                    if (i > 0):
                        ques_locs = answ4ques_locs_var_list[i - 1][0][0]
                        kb_locs = answ4kb_locs_var_list[i - 1][0][0]
                        question_match_count = 0
                        kb_facts_match_count = 0
                        for ques_pos in range(len(ques_locs)):
                            if ques_locs[ques_pos].data[0] == 1:
                                weighted_question_encoding += encoder_outputs[ques_pos][0].view(1, 1, -1)
                                question_match_count += 1
                        if question_match_count > 0:
                            weighted_question_encoding /= question_match_count
                        for kb_idx in range(len(kb_locs)):
                            if kb_locs[kb_idx].data[0] == 1:
                                weighted_kb_facts_encoding += kb_facts_embedded[kb_idx]
                                kb_facts_match_count += 1
                        if kb_facts_match_count > 0:
                            weighted_kb_facts_encoding /= kb_facts_match_count

                    decoder_input_embedded = torch.cat((word_embedded, weighted_question_encoding,
                                                        weighted_kb_facts_encoding, avg_kb_facts_embedded), 2)

                    common_predict, decoder_hidden, mode_predict, kb_atten_predict, hist_kb, ques_atten_predict, hist_ques = self.decoder(
                        word_embedded, decoder_input_embedded, decoder_hidden, question_embedded, kb_facts_embedded,
                        hist_kb, encoder_outputs, hist_ques)

                    ###################### Calculate Loss ################################
                    loss += self.instance_weight * self.mode_loss_rate * XEnLoss(mode_predict.view(1, -1), answer_mode)

                    mode_predict = nn.Softmax(dim=2)(mode_predict).view(3, 1)
                    common_mode_predict = mode_predict[0]
                    kb_mode_predict = mode_predict[1]
                    ques_mode_predict = mode_predict[2]

                    predicted_probs = torch.cat(
                        (common_predict * common_mode_predict, kb_atten_predict * kb_mode_predict,
                         ques_atten_predict * ques_mode_predict), 2)
                    if (answer_mode.data[0] == 0):  # predict mode
                        target = answ_var[i]
                    elif (answer_mode.data[0] == 1):  # retrieve mode
                        kb_locs = answ4kb_locs_var_list[i][0][0]
                        target = self.word_indexer.wordCount + kb_locs.data.tolist().index(1)
                        target = Variable(torch.LongTensor([target]).view(-1))
                        if use_cuda:
                            target = target.cuda()
                    else:  # copy mode
                        ques_locs = answ4ques_locs_var_list[i][0][0]
                        target = self.word_indexer.wordCount + self.max_fact_num + ques_locs.data.tolist().index(1)
                        target = Variable(torch.LongTensor([target]).view(-1))
                        if use_cuda:
                            target = target.cuda()
                    loss += self.instance_weight * XEnLoss(predicted_probs.view(1, -1), target)
                    #####################################################################

                    decoder_input = answ_var[i]
                #####################################################################

                if (iter + 1) % self.batch_size == 0:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                loss = loss.data[0] / answ_length
                lossTotal += loss
                if (iter + 1) % 1000 == 0:
                    lossAvg = lossTotal / 1000
                    lossTotal = 0
                    secs = time.time() - startTime
                    mins = math.floor(secs / 60)
                    secs -= mins * 60
                    print('%dm %ds' % (mins, secs), 'after iteration:', iter + 1, 'with avg loss:', lossAvg)

        self.has_trained = True
        print('Training completed!')

    def predict(self, ques_var, kb_var_list, kb_facts, ques):
        # inputLength = inputVar.size()[0]

        #################### Process KB facts ###############################
        kb_facts_embedded = []
        for rel_obj in kb_var_list:
            rel_embedded = self.embedding(rel_obj[0]).view(1, 1, -1)
            obj_embedded = self.embedding(rel_obj[1]).view(1, 1, -1)
            kb_facts_embedded.append(torch.cat((rel_embedded, obj_embedded), 2))
        avg_kb_facts_embedded = kb_facts_embedded[-1]
        #####################################################################


        ######################### Encoding ##################################
        self.encoder.hidden = self.encoder.init_hidden()
        encoder_outputs = self.encoder(ques_var)
        question_embedded = self.encoder.hidden[0].view(1, 1, -1)
        cell_state = self.encoder.hidden[1].view(1, 1, -1)
        #####################################################################



        ######################### Decoding ###################################
        decoder_hidden = (question_embedded, cell_state)
        decoder_input = Variable(torch.LongTensor([[SOS]]))
        hist_ques = Variable(torch.zeros(1, 1, self.max_ques_len))
        hist_kb = Variable(torch.zeros(1, 1, self.max_fact_num))
        if use_cuda:
            decoder_input = decoder_input.cuda()
            hist_kb = hist_kb.cuda()
            hist_ques = hist_ques.cuda()

        decoded_id = []
        decoded_token = []
        weighted_question_encoding = Variable(torch.zeros(1, 1, 2 * self.state_size))
        weighted_kb_facts_encoding = Variable(torch.zeros(1, 1, 2 * self.embedding_size))
        if use_cuda:
            weighted_question_encoding = weighted_question_encoding.cuda()
            weighted_kb_facts_encoding = weighted_kb_facts_encoding.cuda()
        for i in range(self.MAX_LENGTH):
            word_embedded = self.embedding(decoder_input).view(1, 1, -1)
            decoder_input_embedded = torch.cat((word_embedded, weighted_question_encoding,
                                                weighted_kb_facts_encoding, avg_kb_facts_embedded), 2)

            common_predict, decoder_hidden, mode_predict, kb_atten_predict, hist_kb, ques_atten_predict, hist_ques = self.decoder(
                word_embedded, decoder_input_embedded, decoder_hidden, question_embedded, kb_facts_embedded,
                hist_kb, encoder_outputs, hist_ques)

            mode_predict = nn.Softmax(dim=2)(mode_predict).view(3, 1)
            common_mode_predict = mode_predict[0]
            kb_mode_predict = mode_predict[1]
            ques_mode_predict = mode_predict[2]

            predicted_probs = torch.cat((common_predict * common_mode_predict, kb_atten_predict * kb_mode_predict,
                                         ques_atten_predict * ques_mode_predict), 2)
            topv3, topi3 = predicted_probs.data.topk(3)
            idx = topi3[0][0][0]
            if idx < self.word_indexer.wordCount:  # predict mode
                if idx == EOS:
                    decoded_id.append(EOS)
                    decoded_token.append("_EOS")
                    break
                else:
                    decoded_id.append(idx)
                    word = self.word_indexer.index2word[idx]
                    decoded_token.append(word)
                    decoder_input = Variable(torch.LongTensor([[idx]]))
                    weighted_kb_facts_encoding = Variable(torch.zeros(1, 1, 2 * self.embedding_size))
            elif idx < self.word_indexer.wordCount + self.max_fact_num:  # retrieve mode
                kb_idx = idx - self.word_indexer.wordCount
                rel_obj_idx = kb_var_list[kb_idx]
                obj_idx = rel_obj_idx[1]
                decoded_id.append(obj_idx.data[0])
                kb_sub, kb_rel, kb_obj = kb_facts[kb_idx]
                decoded_token.append(kb_obj)
                decoder_input = obj_idx
                weighted_kb_facts_encoding = kb_facts_embedded[kb_idx]
            else:  # copy mode
                copy_idx = idx - self.word_indexer.wordCount - self.max_fact_num
                word_idx = ques_var[copy_idx]
                decoded_id.append(word_idx.data[0])
                if copy_idx < len(ques):
                    word = ques[copy_idx]
                else:
                    word = FIL
                decoded_token.append(word)
                decoder_input = word_idx
                weighted_question_encoding = encoder_outputs[copy_idx].view(1, 1, -1)
            if use_cuda:
                weighted_kb_facts_encoding = weighted_kb_facts_encoding.cuda()
                decoder_input = decoder_input.cuda()

        return decoded_id, decoded_token










