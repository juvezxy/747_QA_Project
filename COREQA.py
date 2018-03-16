# coding:utf-8

from Encoder import *
from Decoder import *
from DataUtils import *
from config import *

class COREQA(object):

    def __init__(self, model_params):
        self.word_indexer = model_params["word_indexer"]
        self.embedding_size = model_params["embedding_size"]
        self.state_size = model_params["state_size"]
        self.mode_size = model_params["mode_size"]
        self.ques_attention_size = model_params["ques_attention_size"]
        self.kb_attention_size = model_params["kb_attention_size"]
        self.max_fact_num = model_params["max_fact_num"]

        self.learning_rate = model_params["learning_rate"]
        self.L2_factor = model_params["L2_factor"]
        self.MAX_LENGTH = model_params["MAX_LENGTH"]
        self.has_trained = False

        ################ Initialize graph components ########################
        self.embedding = nn.Embedding(self.word_indexer.wordCount, self.embedding_size)
        self.encoder = Encoder(self.word_indexer.wordCount, self.state_size, self.embedding)
        self.decoder = Decoder(output_size=self.word_indexer.wordCount, state_size=self.state_size,
                               embedding=self.embedding, mode_size=self.mode_size,
                               kb_attention_size=self.kb_attention_size, max_fact_num=self.max_fact_num)

        if use_cuda:
            self.encoder.cuda()
            self.decoder.cuda()
        #####################################################################

        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                    lr=self.learning_rate, weight_decay=self.L2_factor)




    def fit(self, training_data):
        if self.has_trained:
            print('Warning! Trying to fit a trained model.')

        print('Start training ...')
        startTime = time.time()

        lossTotal = 0
        criterion = nn.NLLLoss()

        for iter in range(len(training_data)):
            ques_var, answ_var, kb_var_list  = vars_from_data(training_data[iter])
            answ_length = answ_var.size()[0]
            self.optimizer.zero_grad()

            #################### Process KB facts ###############################
            kb_facts_embedded = []
            for rel_obj in kb_var_list:
                rel_embedded = self.embedding(rel_obj[0]).view(1, 1, -1)
                obj_embedded = self.embedding(rel_obj[1]).view(1, 1, -1)
                kb_facts_embedded.append(torch.cat((rel_embedded, obj_embedded), 2))
            avg_kb_facts_embedded = kb_facts_embedded[-1]
            #####################################################################


            ######################### Encoding ###################################
            self.encoder.hidden = self.encoder.init_hidden()

            encoder_outputs = self.encoder(ques_var)
            # pad encodeOutputs to MAX_LENGTH
            #encoder_outputs = encoder_outputs.view(len(encoder_outputs), -1)
            #padding = (0, 0, 0, self.MAX_LENGTH - len(encoder_outputs))
            #encoder_outputs = F.pad(encoder_outputs, padding)
            encoder_outputs = Variable(encoder_outputs)
            if use_cuda:
                encoder_outputs = encoder_outputs.cuda()

            question_embedded = self.encoder.hidden[0].view(1, 1, -1)
            cell_state = self.encoder.hidden[1].view(1, 1, -1)
            #####################################################################


            ######################### Decoding ###################################
            decoder_hidden = (question_embedded, cell_state)
            decoder_input = Variable(torch.LongTensor([[SOS]]))
            hist_kb = Variable(torch.zeros(1, 1, self.max_fact_num))
            if use_cuda:
                decoder_input = decoder_input.cuda()

            loss = 0
            
            for i in range(answ_length):
                word_embedded = self.embedding(decoder_input).view(1, 1, -1)
                question_match_count = 0
                weighted_question_encoding = Variable(torch.zeros(1, 1, 2 * self.state_size))
                kb_facts_match_count = 0
                weighted_kb_facts_encoding = Variable(torch.zeros(1, 1, 2 * self.embedding_size))
                for ques_pos in len(ques_var):
                    if ques_var[ques_pos][0] == decoder_input:
                        weighted_question_encoding += encoder_outputs[ques_pos][0]
                        question_match_count += 1
                if question_match_count > 0:
                    weighted_question_encoding /= question_match_count
                for kb_idx in len(kb_var_list):
                    rel_obj = kb_var_list[kb_idx]
                    if rel_obj[1] == decoder_input:
                        weighted_kb_facts_encoding += kb_facts_embedded[kb_idx][0][0]
                        kb_facts_match_count += 1
                if kb_facts_match_count > 0:
                    weighted_kb_facts_encoding /= kb_facts_match_count
                decoder_input_embedded = torch.cat((word_embedded, weighted_question_encoding,
                                                   weighted_kb_facts_encoding, avg_kb_facts_embedded), 2)

                common_predict, decoder_hidden, mode_predict, kb_atten_predict, hist_kb = self.decoder(word_embedded,
                                                                                               decoder_input_embedded,
                                                                                               decoder_hidden,
                                                                                               question_embedded,
                                                                                               kb_facts_embedded,
                                                                                               hist_kb)





                decoder_input = answ_var[i]

                

            #####################################################################

            loss.backward()

            self.optimizer.step()

            loss =  loss.data[0] / answ_length

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

    def predict(self, inputVar):
        if self.has_trained:
            print('Warning! Trying to predict without training!')

        #inputLength = inputVar.size()[0]

        self.encoder.hidden = self.encoder.initHidden()
        #for i in range(inputLength):
        #    encoderOutput, encoderHidden = self.encoder(inputVar[i], encoderHidden)
        encoderOutputs = self.encoder(inputVar)
        
        decoderInput = Variable(torch.LongTensor([[SOS]]))
        if use_cuda:
            decoderInput = decoderInput.cuda()
        decoderHidden = self.encoder.hidden
        decoded = []
        for i in range(self.MAX_LENGTH):
            decoderOutput, decoderHidden = self.decoder(decoderInput, decoderHidden)
            topv, topi = decoderOutput.data.topk(1)
            token = topi[0][0]
            if token == EOS:
                decoded.append(EOS)
                break
            else:
                decoded.append(token)
            decoderInput = Variable(torch.LongTensor([[token]]))
            if use_cuda:
                decoderInput = decoderInput.cuda()
        return decoded

















