# coding:utf-8

from config import *
from COREQA import *


def evaluate(model, testing_data):
    wordIndexer = model.word_indexer
    if not model.has_trained:
        print('Warning! Trying to evaluate without training!')
    print('Evaluating ...')
    precisionTotal = 0
    recallTotal = 0
    F1Total = 0

    test_length = len(testing_data)
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

    for iter in range(test_length):
        ques_var, answ_var, kb_var_list, answer_modes_var_list, answ4ques_locs_var_list, answ4kb_locs_var_list, kb_facts = vars_from_data(
            testing_data[iter])
        inputSeq, target = testing_data[iter][0], testing_data[iter][1]
        
        target = ''.join(target)
       
        #inputSeq, target = testPairs[iter]
        #ques_var = Variable(torch.LongTensor(inputSeq).view(-1, 1))
        #if use_cuda:
        #    ques_var = ques_var.cuda()
        #predictedSeq = model.predict(ques_var)

        predictedId, predictedToken = model.predict(ques_var, kb_var_list, kb_facts)
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
        if (iter + 1) % 100 == 0:
            predicted = predictedToken
            question = inputSeq
            print('Question:', question)
            print('Gold Answer:', target)
            print('Generated Answer:', predicted)

        #entIndex = inputSeq[0]  # index of the entity
        #entity = wordIndexer.index2word[entIndex]
        entity = inputSeq[0]
        entityNumber = int(re.findall('\d+', entity)[0])
        #predictedEntity = entIndex in predictedSeq
        predictedEntity = False
        predictedMale = wordIndexer.word2index[u'他'] in predictedId
        predictedFemale = wordIndexer.word2index[u'她'] in predictedId
        if predictedEntity or predictedMale or predictedFemale:
            genderPredicted += 1
        if predictedEntity or (predictedMale and entityNumber <= 40000) or (predictedFemale and entityNumber > 40000):
            genderCorrect += 1

        yearMatch = yearPattern.search(target)
        monthMatch = monthPattern.search(target)
        dayMatch = dayPattern.search(target)
        yearPredicted += model.word_indexer.word2index[u'年'] in predictedId
        monthPredicted += model.word_indexer.word2index[u'月'] in predictedId
        dayPredicted += model.word_indexer.word2index[u'日'] in predictedId or wordIndexer.word2index[u'号'] in predictedId

        if (yearMatch):
            yearAppear += 1
            year = yearMatch.group()[:-1]
            if year in predictedToken:
                yearCorrect += 1
        if (monthMatch):
            monthAppear += 1
            month = monthMatch.group()[:-1]
            if month in predictedToken:
                monthCorrect += 1
        if (dayMatch):
            dayAppear += 1
            day = dayMatch.group()[:-1]
            if day in predictedToken:
                dayCorrect += 1

    # print ('Average precision:', precisionTotal / testLength, 'recall:', recallTotal / testLength, 'F1:', F1Total / testLength)
    print('Precision of gender:', genderCorrect * 1.0 / genderPredicted)
    print('Precision of year:', yearCorrect * 1.0 / yearPredicted)
    print('Precision of month:', monthCorrect * 1.0 / monthPredicted)
    print('Precision of day:', dayCorrect * 1.0 / dayPredicted)
    print('Precision:', (genderCorrect + yearCorrect + monthCorrect + dayCorrect) * 1.0 / (
    genderPredicted + yearPredicted + monthPredicted + dayPredicted))
    #print('Recall:', (genderCorrect + yearCorrect + monthCorrect + dayCorrect) * 1.0 / (testLength + yearAppear + monthAppear + dayAppear))