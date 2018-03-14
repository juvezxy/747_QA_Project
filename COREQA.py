# coding:utf-8

from Encoder import *
from Decoder import *
from DataUtils import *
from config import *

class COREQA(object):

    def __init__(self, model_params):
        self.wordIndexer = model_params["wordIndexer"]
        self.embeddingSize = model_params["embeddingSize"]
        self.learningRate = model_params["learningRate"]
        self.MAX_LENGTH = model_params["MAX_LENGTH"]
        self.hasTrained = False

        self.encoder = Encoder(self.wordIndexer.wordCount, self.embeddingSize)
        self.decoder = Decoder(self.wordIndexer.wordCount, self.embeddingSize)
        self.encoderOptimizer = optim.Adam(self.encoder.parameters(), lr=self.learningRate)
        self.decoderOptimizer = optim.Adam(self.decoder.parameters(), lr=self.learningRate)

        if use_cuda:
            self.encoder.cuda()
            self.decoder.cuda()


    def fit(self, pairs):
        if self.hasTrained:
            print('Warning! Trying to fit a trained model.')

        print('Start training ...')
        startTime = time.time()
        lossTotal = 0

        criterion = nn.NLLLoss()

        for iter in range(len(pairs)):
            inputVar, targetVar = varsFromPair(pairs[iter])
            inputLength = inputVar.size()[0]
            targetLength = targetVar.size()[0]

            self.encoderOptimizer.zero_grad()
            self.decoderOptimizer.zero_grad()

            encoderHidden = self.encoder.initHidden()
            for i in range(inputLength):
                encoderOutput, encoderHidden = self.encoder(inputVar[i], encoderHidden)

            decoderInput = Variable(torch.LongTensor([[SOS]]))
            if use_cuda:
                decoderInput = decoderInput.cuda()
            decoderHidden = encoderHidden
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

        self.hasTrained = True
        print('Training completed!')

    def predict(self, inputVar):
        if self.hasTrained:
            print('Warning! Trying to predict without training!')

        inputLength = inputVar.size()[0]

        encoderHidden = self.encoder.initHidden()
        for i in range(inputLength):
            encoderOutput, encoderHidden = self.encoder(inputVar[i], encoderHidden)

        decoderInput = Variable(torch.LongTensor([[SOS]]))
        if use_cuda:
            decoderInput = decoderInput.cuda()
        decoderHidden = encoderHidden
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

















