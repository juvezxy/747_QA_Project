# coding:utf-8

from config import *
from COREQA import *


def evaluate(model, testing_data):
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

    totalPrecision = 0
    totalRecall = 0
    for iter in range(test_length):
        ques_var, answ_var, kb_var_list, answer_modes_var_list, answ4ques_locs_var_list, answ4kb_locs_var_list, kb_facts, ques, answ = vars_from_data(
            testing_data[iter])
        target = ''.join(answ)
       
        #inputSeq, target = testPairs[iter]
        #ques_var = Variable(torch.LongTensor(inputSeq).view(-1, 1))
        #if use_cuda:
        #    ques_var = ques_var.cuda()
        #predictedSeq = model.predict(ques_var)

        predictedId, predictedToken = model.predict(ques_var, kb_var_list, kb_facts, ques)
        predictedSent = ''.join(predictedToken)
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
        if (iter + 1) % 1000 == 0:
            print(repr(ques).decode('unicode-escape'))
            print(repr(answ).decode('unicode-escape'))
            print(repr(predictedToken).decode('unicode-escape'))

        predictedCount = 0
        appearCount = 0
        correctCount = 0

        #entIndex = inputSeq[0]  # index of the entity
        #entity = wordIndexer.index2word[entIndex]
        entity = ques[0]
        entityNumber = int(re.findall('\d+', entity)[0])
        #predictedEntity = entIndex in predictedSeq
        predictedEntity = ques[0] in predictedToken
        predictedMale = u'他' in predictedToken
        predictedFemale = u'她' in predictedToken
        if predictedEntity or predictedMale or predictedFemale:
            genderPredicted += 1
            predictedCount += 1
        appearCount += 1
        if predictedEntity or (predictedMale and entityNumber <= 40000) or (predictedFemale and entityNumber > 40000):
            genderCorrect += 1
            correctCount += 1


        yearMatchTarget = yearPattern.search(target)
        monthMatchTarget = monthPattern.search(target)
        dayMatchTarget = dayPattern.search(target)

        yearMatchPredicted = yearPattern.search(predictedSent)
        monthMatchPredicted = monthPattern.search(predictedSent)
        dayMatchPredicted = dayPattern.search(predictedSent)

        if (yearMatchPredicted):
            yearPredicted += 1
            predictedCount += 1
            yearPredict = yearMatchPredicted.group()[:-1]
        if (yearMatchTarget):
            yearAppear += 1
            appearCount += 1
            year = yearMatchTarget.group()[:-1]
            if yearMatchPredicted and year == yearPredict:
                yearCorrect += 1
                correctCount += 1

        if (monthMatchPredicted):
            monthPredicted += 1
            predictedCount += 1
            monthPredict = monthMatchPredicted.group()[:-1]
        if (monthMatchTarget):
            monthAppear += 1
            appearCount += 1
            month = monthMatchTarget.group()[:-1]
            if monthMatchPredicted and month == monthPredict:
                monthCorrect += 1
                correctCount += 1

        if (dayMatchPredicted):
            dayPredicted += 1
            predictedCount += 1
            dayPredict = dayMatchPredicted.group()[:-1]
        if (dayMatchTarget):
            dayAppear += 1
            appearCount += 1
            day = dayMatchTarget.group()[:-1]
            if dayMatchPredicted and day == dayPredict:
                dayCorrect += 1
                correctCount += 1

        if correctCount == predictedCount:
            totalPrecision += 1
        totalRecall += correctCount * 1.0 / appearCount
    # print ('Average precision:', precisionTotal / testLength, 'recall:', recallTotal / testLength, 'F1:', F1Total / testLength)
    if genderPredicted > 0:
        print('Precision of gender:', genderCorrect * 1.0 / genderPredicted)
        #print (genderCorrect)
    print('Recall of gender:', genderCorrect * 1.0 / test_length)
    if yearPredicted > 0:
        print('Precision of year:', yearCorrect * 1.0 / yearPredicted)
        #print (yearCorrect)
    if yearAppear > 0:
        print('Recall of year:', yearCorrect * 1.0 / yearAppear)
    if monthPredicted > 0:
        print('Precision of month:', monthCorrect * 1.0 / monthPredicted)
        #print (monthCorrect)
    if monthAppear > 0:
        print('Recall of month:', monthCorrect * 1.0 / monthAppear)
    if dayPredicted > 0:
        print('Precision of day:', dayCorrect * 1.0 / dayPredicted)
        #print (dayCorrect)
    if dayAppear > 0:
        print('Recall of day:', dayCorrect * 1.0 / dayAppear)
    print('Precision across 4 categories:', (genderCorrect + yearCorrect + monthCorrect + dayCorrect) * 1.0 / (genderPredicted + yearPredicted + monthPredicted + dayPredicted))
    print('Recall across 4 categories:', (genderCorrect + yearCorrect + monthCorrect + dayCorrect) * 1.0 / (test_length + yearAppear + monthAppear + dayAppear))
    print('Total precision:', totalPrecision * 1.0 / test_length)
    print('Total recall:', totalRecall * 1.0 / test_length)
