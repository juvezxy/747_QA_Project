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
        self.learning_rate = model_params["learning_rate"]
        self.MAX_LENGTH = model_params["MAX_LENGTH"]
        self.has_trained = False

        self.embedding = nn.Embedding(self.word_indexer.wordCount, self.embedding_size)
        self.encoder = Encoder(self.word_indexer.wordCount, self.state_size)
        self.decoder = Decoder(self.word_indexer.wordCount, self.state_size)
        self.encoderOptimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoderOptimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)

        if use_cuda:
            self.encoder.cuda()
            self.decoder.cuda()


    def fit(self, training_data):
        if self.has_trained:
            print('Warning! Trying to fit a trained model.')

        print('Start training ...')
        startTime = time.time()
        lossTotal = 0

        criterion = nn.NLLLoss()

        for iter in range(len(training_data)):
            inputVar, targetVar = varsFromPair(pairs[iter])
            #inputLength = inputVar.size()[0]
            targetLength = targetVar.size()[0]

            self.encoderOptimizer.zero_grad()
            self.decoderOptimizer.zero_grad()

            self.encoder.hidden = self.encoder.initHidden()
            #for i in range(inputLength):
            #    encoderOutput, encoderHidden = self.encoder(inputVar[i], encoderHidden)
            encoderOutputs = self.encoder(inputVar)
            # pad encodeOutputs to MAX_LENGTH
            encoderOutputs = encoderOutputs.view(len(encoderOutputs), -1)
            padding = (0, 0, 0, self.MAX_LENGTH - len(encoderOutputs))
            encoderOutputs = F.pad(encoderOutputs, padding)
            encoderOutputs = Variable(encoderOutputs)
            if use_cuda:
                encoderOutputs = encoderOutputs.cuda()

            decoderInput = Variable(torch.LongTensor([[SOS]]))
            if use_cuda:
                decoderInput = decoderInput.cuda()
            decoderHidden = self.encoder.hidden
            loss = 0

            for i in range(targetLength):
                decoderOutput, decoderHidden = self.decoder(decoderInput, decoderHidden)
                loss += criterion(decoderOutput, targetVar[i])
                decoderInput = targetVar[i]
            loss.backward()

            self.encoderOptimizer.step()
            self.decoderOptimizer.step()
            loss =  loss.data[0] / targetLength

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

















