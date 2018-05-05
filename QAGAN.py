# coding:utf-8

from Encoder import *
from Decoder import *
from Positioner import *
from Discriminator import *
from DataUtilsNew import *
from config import *
import helpers
import sys

class QAGAN(object):
    def __init__(self, model_params):
        self.word_indexer = model_params["word_indexer"]
        self.word_embedder = model_params["word_embedder"]
        self.embedding_size = model_params["embedding_size"]
        self.state_size = model_params["state_size"]
        self.mode_size = model_params["mode_size"]
        self.position_size = model_params["position_size"]
        self.ques_attention_size = model_params["ques_attention_size"]
        self.kb_attention_size = model_params["kb_attention_size"]
        self.dis_embedding_dim = model_params["dis_embedding_dim"]
        self.dis_hidden_dim = model_params["dis_hidden_dim"]
        self.max_fact_num = model_params["max_fact_num"]
        self.max_ques_len = model_params["max_ques_len"]

        self.learning_rate = model_params["learning_rate"]
        self.mode_loss_rate = model_params["mode_loss_rate"]
        self.position_loss_rate = model_params["position_loss_rate"]
        self.L2_factor = model_params["L2_factor"]
        self.batch_size = model_params["batch_size"]
        self.adv_batch_size = model_params["adv_batch_size"]
        self.epoch_size = model_params["epoch_size"]
        self.adv_epoch_size = model_params["adv_epoch_size"]
        self.instance_weight = 1.0 / self.batch_size
        self.MAX_LENGTH = model_params["MAX_LENGTH"]
        self.has_trained = False
        self.oracle_samples = []
        self.negative_samples = torch.zeros(1, self.MAX_LENGTH, self.embedding_size)

        ################ Initialize graph components ########################
        self.encoder = Encoder(self.word_indexer.wordCount, self.state_size, self.embedding_size)
        self.decoder = Decoder(output_size=self.word_indexer.wordCount, state_size=self.state_size,
                               embedding_size=self.embedding_size, mode_size=self.mode_size,
                               kb_attention_size=self.kb_attention_size, max_fact_num=self.max_fact_num,
                               ques_attention_size=self.ques_attention_size, max_ques_len=self.max_ques_len, position_size=self.MAX_LENGTH)
        self.positioner = Positioner(self.state_size, self.position_size, self.MAX_LENGTH)
        self.dis = Discriminator(self.embedding_size, self.dis_hidden_dim, self.word_indexer.wordCount, self.MAX_LENGTH, gpu=use_cuda)

        if use_cuda:
            self.encoder.cuda()
            self.decoder.cuda()
            self.positioner.cuda()
            self.dis.cuda()
            self.negative_samples = self.negative_samples.cuda()

        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.positioner.parameters()),
                                    lr=self.learning_rate, weight_decay=self.L2_factor)
        self.dis_optimizer = optim.Adagrad(self.dis.parameters())
        #####################################################################

    def fit(self, training_data, policy_gradient):
        if self.has_trained:
            print('Warning! Trying to fit a trained model.')

        epochs = self.adv_epoch_size if policy_gradient else self.epoch_size

        if policy_gradient:
            print('Start policy gradient training ...')
        else:
            print('Start training ...')
            oracle_samples = []
            for iter in range(len(training_data)):
                ques_var, answ_var, kb_var, kb_position_var, answ_id_var, answer_modes_var_list, answ4ques_locs_var_list, answ4kb_locs_var_list, kb_facts, ques, answ = vars_from_data(
                    training_data[iter])
                padding = self.MAX_LENGTH - answ_var.size()[0]
                if (padding >= 0):
                    answ_var = F.pad(answ_var, (0, 0, 0, 0, 0, padding), "constant", 0)
                    oracle_samples.append(answ_var.permute(1,0,2))
            self.oracle_samples = torch.cat(oracle_samples, 0).data
        startTime = time.time()
        XEnLoss = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            lossTotal = 0.0
            self.optimizer.zero_grad()
            shuffle(training_data)
            loss = 0.0
            training_len = len(training_data)
            negative_samples = []
            for iter in range(training_len):
                ques_var, answ_var, kb_var, kb_position_var, answ_id_var, answer_modes_var_list, answ4ques_locs_var_list, answ4kb_locs_var_list, kb_facts, ques, answ = vars_from_data(
                    training_data[iter])
                answ_length = answ_var.size()[0]

                #################### Process KB facts ###############################
                kb_fact_embedded = kb_var
                #####################################################################


                ######################### Encoding ##################################
                self.encoder.hidden = self.encoder.init_hidden()
                encoder_outputs = self.encoder(ques_var)
                question_embedded = self.encoder.hidden[0].view(1, 1, -1)
                cell_state = self.encoder.hidden[1].view(1, 1, -1)
                #####################################################################

                ######################### Position ##################################

                position_predict = self.positioner(question_embedded)

                #####################################################################


                ######################### Decoding ###################################
                decoder_hidden = (question_embedded, cell_state)
                decoder_input = Variable(SOS_NUMPY)
                hist_ques = Variable(torch.zeros(1, 1, self.max_ques_len))
                hist_kb = Variable(torch.zeros(1, 1, self.max_fact_num))
                if use_cuda:
                    decoder_input = decoder_input.cuda()
                    hist_kb = hist_kb.cuda()
                    hist_ques = hist_ques.cuda()

                decoded_seq = []
                cond_probs = []
                seq_len = self.MAX_LENGTH if policy_gradient else answ_length
                for i in range(seq_len):
                    if (not policy_gradient):
                        answer_mode = answer_modes_var_list[i]
                    word_embedded = decoder_input
                    weighted_question_encoding = Variable(torch.zeros(1, 1, 2 * self.state_size))
                    weighted_kb_facts_encoding = Variable(torch.zeros(1, 1, self.embedding_size))
                    if use_cuda:
                        weighted_question_encoding = weighted_question_encoding.cuda()
                        weighted_kb_facts_encoding = weighted_kb_facts_encoding.cuda()

                    if (not policy_gradient) and (i > 0):
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
                                weighted_kb_facts_encoding += kb_fact_embedded
                                kb_facts_match_count += 1
                        if kb_facts_match_count > 0:
                            weighted_kb_facts_encoding /= kb_facts_match_count
                    decoder_input_embedded = torch.cat((word_embedded, weighted_question_encoding,
                                                        weighted_kb_facts_encoding, position_predict), 2)

                    common_predict, decoder_hidden, mode_predict, kb_atten_predict, hist_kb, ques_atten_predict, hist_ques = self.decoder(
                        word_embedded, decoder_input_embedded, decoder_hidden, question_embedded, kb_fact_embedded,
                        hist_kb, encoder_outputs, hist_ques)

                        
                    if policy_gradient:
                        ###################### Generate next token ################################

                        mode_predict = nn.Softmax(dim=2)(mode_predict).view(3, 1)
                        common_mode_predict = mode_predict[0]
                        kb_mode_predict = mode_predict[1]
                        ques_mode_predict = mode_predict[2]

                        predicted_probs = torch.cat((common_predict * common_mode_predict, kb_atten_predict * kb_mode_predict,
                                                     ques_atten_predict * ques_mode_predict), 2)
                        topv3, topi3 = predicted_probs.data.topk(3)
                        idx = topi3[0][0][0]
                        cond_probs.append(topv3[0][0][0])
                        if idx < self.word_indexer.wordCount:  # predict mode
                            if idx == EOS:
                                decoded_seq.append(Variable(EOS_NUMPY).cuda())
                                break
                            else:
                                word = self.word_indexer.index2word[idx]
                                decoder_input = Variable(self.word_embedder[word])
                                decoded_seq.append(decoder_input.cuda())
                                weighted_kb_facts_encoding = Variable(torch.zeros(1, 1, self.embedding_size))
                        elif idx < self.word_indexer.wordCount + self.max_fact_num:  # retrieve mode
                            decoder_input = kb_var
                            decoded_seq.append(decoder_input.cuda())
                            weighted_kb_facts_encoding = kb_fact_embedded
                        else:  # copy mode
                            copy_idx = idx - self.word_indexer.wordCount - self.max_fact_num
                            word_idx = ques_var[copy_idx]
                            if copy_idx < len(ques):
                                word = ques[copy_idx]
                            else:
                                word = FIL
                            decoder_input = word_idx.view(1,1,-1).narrow(2, 0, 1024)
                            decoded_seq.append(decoder_input.cuda())
                            weighted_question_encoding = encoder_outputs[copy_idx].view(1, 1, -1)
                        if use_cuda:
                            weighted_kb_facts_encoding = weighted_kb_facts_encoding.cuda()
                            decoder_input = decoder_input.cuda()
                       
                        #####################################################################
                    else:
                        ###################### Calculate Loss ################################

                        loss += self.instance_weight * self.mode_loss_rate * XEnLoss(mode_predict.view(1, -1), answer_mode)
                        loss += self.instance_weight * self.position_loss_rate * XEnLoss(position_predict.view(1, -1), kb_position_var)

                        mode_predict = nn.Softmax(dim=2)(mode_predict).view(3, 1)
                        common_mode_predict = mode_predict[0]
                        kb_mode_predict = mode_predict[1]
                        ques_mode_predict = mode_predict[2]

                        predicted_probs = torch.cat(
                            (common_predict * common_mode_predict, kb_atten_predict * kb_mode_predict,
                             ques_atten_predict * ques_mode_predict), 2)
                        if (answer_mode.data[0] == 0):  # predict mode
                            target = answ_id_var[i]
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
                        decoder_input = answ_var[i].view(1,1,-1)
                        
                        #####################################################################
                
                #####################################################################
                if policy_gradient:
                    decoded_seq = torch.cat(decoded_seq, 0)
                    padding = self.MAX_LENGTH - decoded_seq.size()[0]
                    if (padding >= 0):
                        decoded_seq = F.pad(decoded_seq, (0, 0, 0, 0, 0, padding), "constant", 0)
                    if use_cuda:
                        decoded_seq = decoded_seq.cuda()
                    negative_samples.append(decoded_seq.permute(1,0,2))
                    rewards = self.dis.batchClassify(decoded_seq)

                    for i in range(len(cond_probs)):
                        loss += math.log(cond_probs[i]) * rewards
                    if (iter + 1) % self.adv_batch_size == 0:

                        loss.sum().backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        loss = loss.data[0]
                        lossTotal += loss
                        loss = 0.0
                else:
                    if (iter + 1) % self.batch_size == 0:
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        loss = loss.data[0]
                        lossTotal += loss
                        loss = 0.0

                    if (iter + 1) % 10 == 0:
                        lossAvg = lossTotal / 10
                        lossTotal = 0
                        secs = time.time() - startTime
                        mins = math.floor(secs / 60)
                        secs -= mins * 60
                        print('%dm %ds' % (mins, secs), 'after iteration:', iter + 1, 'with avg loss:', lossAvg)

            ###################### Train discriminator ################################
            if policy_gradient:
                self.negative_samples = torch.cat(negative_samples, 0).data
                print('Training discriminator ...')
                self.train_discriminator(10, 3)
        if (not policy_gradient):
            self.has_trained = True
            print('Training completed!')
            print('Pretraining discriminator ...')
            self.train_discriminator(50, 3)

    def train_discriminator(self, d_steps, epochs):
        for d_step in range(d_steps):
            dis_inp, dis_target = helpers.prepare_discriminator_data(self.oracle_samples, self.negative_samples, gpu=use_cuda)
            for epoch in range(epochs):
                print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
                sys.stdout.flush()
                total_loss = 0
                total_acc = 0

                samples = self.oracle_samples.size()[0] + self.negative_samples.size()[0]
                for i in range(0, samples, self.adv_batch_size):
                    inp, target = dis_inp[i:i + self.adv_batch_size], dis_target[i:i + self.adv_batch_size]
                    self.dis_optimizer.zero_grad()
                    out = self.dis.batchClassify(inp)
                    loss_fn = nn.BCELoss()
                    loss = loss_fn(out, target)
                    loss.backward()
                    self.dis_optimizer.step()

                    total_loss += loss.data[0]
                    total_acc += torch.sum((out>0.5)==(target>0.5)).data[0]

                    if (i / self.adv_batch_size) % math.ceil(math.ceil(samples / float(
                            self.adv_batch_size)) / 10.) == 0:  # roughly every 10% of an epoch
                        print('.', end='')
                        sys.stdout.flush()

                total_loss /= math.ceil(samples / float(self.adv_batch_size))
                total_acc /= float(samples)

    def predict(self, ques_var, kb_var, kb_facts, ques):

        #################### Process KB facts ###############################
        kb_fact_embedded = kb_var

        #####################################################################


        ######################### Encoding ##################################
        self.encoder.hidden = self.encoder.init_hidden()
        encoder_outputs = self.encoder(ques_var)
        question_embedded = self.encoder.hidden[0].view(1, 1, -1)
        cell_state = self.encoder.hidden[1].view(1, 1, -1)
        #####################################################################

        ######################### Position ##################################

        position_predict = self.positioner(question_embedded)

        #####################################################################


        ######################### Decoding ###################################
        decoder_hidden = (question_embedded, cell_state)
        decoder_input = Variable(SOS_NUMPY)
        hist_ques = Variable(torch.zeros(1, 1, self.max_ques_len))
        hist_kb = Variable(torch.zeros(1, 1, self.max_fact_num))
        if use_cuda:
            decoder_input = decoder_input.cuda()
            hist_kb = hist_kb.cuda()
            hist_ques = hist_ques.cuda()

        decoded_id = []
        decoded_token = []
        weighted_question_encoding = Variable(torch.zeros(1, 1, 2 * self.state_size))
        weighted_kb_facts_encoding = Variable(torch.zeros(1, 1, self.embedding_size))
        if use_cuda:
            weighted_question_encoding = weighted_question_encoding.cuda()
            weighted_kb_facts_encoding = weighted_kb_facts_encoding.cuda()
        for i in range(self.MAX_LENGTH):
            word_embedded = decoder_input
            decoder_input_embedded = torch.cat((word_embedded, weighted_question_encoding,
                                                weighted_kb_facts_encoding, position_predict), 2)

            common_predict, decoder_hidden, mode_predict, kb_atten_predict, hist_kb, ques_atten_predict, hist_ques = self.decoder(
                word_embedded, decoder_input_embedded, decoder_hidden, question_embedded, kb_fact_embedded,
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
                    decoded_id.append(Variable(EOS_NUMPY))
                    decoded_token.append("_EOS")
                    break
                else:
                    word = self.word_indexer.index2word[idx]
                    decoded_token.append(word)
                    decoder_input = Variable(self.word_embedder[word])
                    decoded_id.append(decoder_input)
                    weighted_kb_facts_encoding = Variable(torch.zeros(1, 1, self.embedding_size))
            elif idx < self.word_indexer.wordCount + self.max_fact_num:  # retrieve mode
                kb_idx = idx - self.word_indexer.wordCount
                kb_sub, kb_rel, kb_obj = kb_facts[kb_idx]
                decoded_token.append(kb_obj)
                decoder_input = kb_var
                decoded_id.append(decoder_input)
                weighted_kb_facts_encoding = kb_fact_embedded
            else:  # copy mode
                copy_idx = idx - self.word_indexer.wordCount - self.max_fact_num
                word_idx = ques_var[copy_idx]
                if copy_idx < len(ques):
                    word = ques[copy_idx]
                else:
                    word = FIL

                decoded_token.append(word)
                decoder_input = word_idx.view(1,1,-1).narrow(2, 0, 1024)
                decoded_id.append(decoder_input)
                weighted_question_encoding = encoder_outputs[copy_idx].view(1, 1, -1)
            if use_cuda:
                weighted_kb_facts_encoding = weighted_kb_facts_encoding.cuda()
                decoder_input = decoder_input.cuda()

        return decoded_id, decoded_token










