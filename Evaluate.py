# coding:utf-8

from config import *
from COREQA import *

def evaluate(model, data_loader):
    print('Evaluating ...')
    precisionTotal = 0
    recallTotal = 0
    F1Total = 0
    testLength = len(data_loader.testing_data)
    wordIndexer = data_loader.wordIndexer
    genderCorrect = 0
    genderPredicted = 0
    yearCorrect = 0
    monthCorrect = 0
    dayCorrect = 0
    yearPredicted = 0
    monthPredicted = 0
    dayPredicted = 0
    yearAppear = 0
    monthAppear = 0
    dayAppear = 0
    for i in range(testLength):
        ques_var, answ_var, kb_var_list, answer_modes_var_list, answ4ques_locs_var_list, answ4kb_locs_var_list = vars_from_data(
            data_loader.testing_data[i])

        inputVar = Variable(torch.LongTensor(inputSeq).view(-1, 1))
        if use_cuda:
            inputVar = inputVar.cuda()
        predictedSeq = model.predict(inputVar)
        '''precision = precision_score(targetSeq, predictedSeq, average='micro')
        recall = recall_score(targetSeq, predictedSeq, average='micro')
        F1 = 2 * (precision * recall) / (precision + recall)
        precisionTotal += precision
        recallTotal += recall
        F1Total += F1
        if (i+1) % 1000 == 0:
            print ('Test size so far:', i, 'precision:', precisionTotal / (i+1), 'recall:', recallTotal / (i+1),
                'F1:', F1Total / (i+1))
        '''
        if (i + 1) % 1000 == 0:
            predicted = [wordIndexer.index2word[index] for index in predictedSeq]
            question = [wordIndexer.index2word[index] for index in inputSeq]
            print('Question:', question)
            print('Gold Answer:', target)
            print('Generated Answer:', predicted)

        entIndex = inputSeq[0]  # index of the entity
        entity = wordIndexer.index2word[entIndex]
        entityNumber = int(re.findall('\d+', entity)[0])
        predictedEntity = entIndex in predictedSeq
        predictedMale = wordIndexer.word2index['他'] in predictedSeq
        predictedFemale = wordIndexer.word2index['她'] in predictedSeq
        if predictedEntity or predictedMale or predictedFemale:
            genderPredicted += 1
        if predictedEntity or (predictedMale and entityNumber <= 40000) or (predictedFemale and entityNumber > 40000):
            genderCorrect += 1

        yearMatch = yearPattern.search(target)
        monthMatch = monthPattern.search(target)
        dayMatch = dayPattern.search(target)
        yearPredicted += wordIndexer.word2index['年'] in predictedSeq
        monthPredicted += wordIndexer.word2index['月'] in predictedSeq
        dayPredicted += wordIndexer.word2index['日'] in predictedSeq or wordIndexer.word2index['号'] in predictedSeq

        if (yearMatch):
            yearAppear += 1
            year = yearMatch.group()[:-1]
            if wordIndexer.word2index[year] in predictedSeq:
                yearCorrect += 1
        if (monthMatch):
            monthAppear += 1
            month = monthMatch.group()[:-1]
            if wordIndexer.word2index[month] in predictedSeq:
                monthCorrect += 1
        if (dayMatch):
            dayAppear += 1
            day = dayMatch.group()[:-1]
            if wordIndexer.word2index[day] in predictedSeq:
                dayCorrect += 1

    # print ('Average precision:', precisionTotal / testLength, 'recall:', recallTotal / testLength, 'F1:', F1Total / testLength)
    print('Precision of gender:', genderCorrect * 1.0 / genderPredicted)
    print('Precision of year:', yearCorrect * 1.0 / yearPredicted)
    print('Precision of month:', monthCorrect * 1.0 / monthPredicted)
    print('Precision of day:', dayCorrect * 1.0 / dayPredicted)
    print('Precision:', (genderCorrect + yearCorrect + monthCorrect + dayCorrect) * 1.0 / (
    genderPredicted + yearPredicted + monthPredicted + dayPredicted))
    print('Recall:', (genderCorrect + yearCorrect + monthCorrect + dayCorrect) * 1.0 / (
    testLength + yearAppear + monthAppear + dayAppear))