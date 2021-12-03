#AI final project

import math

#read from data file line by line into a list
faceFile = open(r"C:\Users\atfan\OneDrive\Desktop\AI\Final Project\facedata\facedatavalidation",'r')
lines = faceFile.readlines()
faceFile.close()
#read from data labels file label by label into a list
labelFile = open(r"C:\Users\atfan\OneDrive\Desktop\AI\Final Project\facedata\facedatavalidationlabels",'r')
labels = []
for c in labelFile.read():
    if c != '\n' and c != ' ':
        labels.append(c)
labelFile.close()

#Global Variables
DATA_TOTAL = len(labels)
FEATURE_ROWS = 5
FEATURE_COLS = 5
FEATURE_TOTAL = FEATURE_ROWS * FEATURE_COLS
IMAGE_PIX_WIDTH = 60
IMAGE_PIX_HEIGHT = 70
PIX_INCREMENT_W = IMAGE_PIX_WIDTH//FEATURE_ROWS
PIX_INCREMENT_H = IMAGE_PIX_HEIGHT//FEATURE_COLS

class Features:
    def __init__(self, l_list, f_rows, f_cols):
        self.data_total = len(l_list)
        self.startFeatx = 0
        self.endFeatx = 14
        self.list = []
        self.f_rows = f_rows
        self.f_cols = f_cols

    def getFeatures(self, f_list):

        numFeat = 1

        for rows in range(self.f_rows):
            startFeaty = 0
            endFeaty = 12
            for cols in range(self.f_cols):
                count = 0
                for line in lines[self.startFeatx:self.endFeatx]:
                    startFeaty = startFeaty
                    endFeaty = endFeaty
                    for char in line[startFeaty:endFeaty]:
                        #print(char,end = '')
                        if char == '#':
                            count += 1
                    #print('\n')
                #print(numFeat,': ',count)
                f_list.append(count)
                startFeaty += 12
                endFeaty += 12
                numFeat += 1
            self.startFeatx += 14
            self.endFeatx += 14

    def getList(self):
        for x in range(self.data_total):
            featList = []
            self.getFeatures(featList)
            self.list.append(featList)

        return self.list

class NaiveBayes:
    def __init__(self, l_list, f_rows, f_cols, pix_incr_W, pix_incr_H, f_total):
        self.labels_list = l_list
        self.data_total = len(l_list)
        #self.features = Features(l_list, f_rows, f_cols, pix_incr_W, pix_incr_H)
        self.feat_list_2d = [] #self.features.getList()
        self.feat_total = f_total
        self.total_true = self.count_true_total()
        self.total_false = self.data_total - self.total_true
        self.f_rows = f_rows
        self.f_cols = f_cols
        self.pixel_incr_W = pix_incr_W
        self.pixel_incr_H = pix_incr_H
        self.startFeatx = 0
        self.endFeatx = pix_incr_H
        self.true_data_table = []
        self.false_data_table = []
        #self.max_value = 0
        self.true_data_probs_table = []
        self.false_data_probs_table = []
        self.true_prob = self.total_true/self.data_total
        self.false_prob = 1-self.true_prob

    def count_true_total(self):
        total_true = 0
        for x in range(self.data_total):
            if self.labels_list[x] == '1':
                total_true += 1
        return total_true

    def get_features(self, f_list):
        numFeat = 1
        for rows in range(self.f_rows):
            startFeaty = 0
            endFeaty = self.pixel_incr_W
            for cols in range(self.f_cols):
                count = 0
                for line in lines[self.startFeatx:self.endFeatx]:
                    startFeaty = startFeaty
                    endFeaty = endFeaty
                    for char in line[startFeaty:endFeaty]:
                        if char == '#':
                            count += 1
                f_list.append(count)
                startFeaty += self.pixel_incr_W
                endFeaty += self.pixel_incr_W
                numFeat += 1
            self.startFeatx += self.pixel_incr_H
            self.endFeatx += self.pixel_incr_H

    def get_feat_list_2d(self):
        for x in range(self.data_total):
            featList = []
            self.get_features(featList)
            self.feat_list_2d.append(featList)
        return self.feat_list_2d

    def get_data_tables(self):
        self.get_feat_list_2d()
        for y in range(self.feat_total):
            true_rows = []
            false_rows = []
            for x in range(self.data_total):
                if self.labels_list[x] == '1':
                    true_rows.append(self.feat_list_2d[x][y])
                else:
                    false_rows.append(self.feat_list_2d[x][y])
            self.true_data_table.append(true_rows)
            self.false_data_table.append(false_rows)

    def get_results(self):
        self.get_data_tables()
        max_list = []
        for x in self.true_data_table:
            max_list.append(max(x))

        for x in self.false_data_table:
            max_list.append(max(x))

        max_value = max(max_list)

        for i in range(self.feat_total):
            data_count = []
            for num in range(max_value+1):
                numCount = 0
                for j in self.true_data_table[i]:
                    if j == num:
                        numCount += 1
                if numCount != 0:
                    data_count.append(numCount/self.total_true)
                else:
                    data_count.append(0.0001)

            self.true_data_probs_table.append(data_count)

        for i in range(self.feat_total):
            data_count = []
            for num in range(max_value+1):
                numCount = 0
                for j in self.false_data_table[i]:
                    if j == num:
                        numCount += 1
                if numCount != 0:
                    data_count.append(numCount/self.total_false)
                else:
                    data_count.append(0.0001)

            self.false_data_probs_table.append(data_count)

        result_list = []
        for z in range(self.data_total):
            feat_p_t = 0
            for n in range(self.feat_total):
                for check in range(max_value+1):
                    if self.feat_list_2d[z][n] == check:
                        feat_p_t += math.log(self.true_data_probs_table[n][check])

            feat_p_f = 0
            for n in range(self.feat_total):
                for check in range(max_value+1):
                    if self.feat_list_2d[z][n] == check:
                        feat_p_f += math.log(self.false_data_probs_table[n][check])

            true = feat_p_t*self.true_prob
            false = feat_p_f*self.false_prob

            if true > false:
                result_list.append(1)
                #print(z+1,': ','face = True')
            else:
                result_list.append(0)
                #print(z+1,': ','face = False')

        return result_list

def print_accuracy(label_list, d_total):
    classifier = NaiveBayes(labels, FEATURE_ROWS, FEATURE_COLS, PIX_INCREMENT_W, PIX_INCREMENT_H, FEATURE_TOTAL)
    results_list = classifier.get_results()

    correct = 0
    for val in range(d_total):
        if results_list[val] == labels[val]:
            correct += 1

    accuracy = correct/d_total

    print('accuracy: ',round(accuracy*100,2),'%')

def main():

    # classifier = NaiveBayes(labels, FEATURE_ROWS, FEATURE_COLS, PIX_INCREMENT_W, PIX_INCREMENT_H, FEATURE_TOTAL)
    # classifier.get_data_probs_tables()
    # print(classifier.total_true, ',', classifier.total_false)
    # print(classifier.true_prob, ',', classifier.false_prob)
    # for i in range(DATA_TOTAL):
    #     labels[i] = int(labels[i])
    # print_accuracy(labels, DATA_TOTAL)
    #print(classifier.get_results())
    # c = 1
    # for t in classifier.true_data_probs_table:
    #     print(c,': ',sum(t))
    #     c+=1
    # nc = 1
    # for f in classifier.false_data_probs_table:
    #     print(nc,': ',sum(f))
    #     nc+=1

    features = Features(labels, FEATURE_ROWS, FEATURE_COLS)
    feat_list = features.getList()

    # lcount = 1
    # for feat in feat_list:
    #     print('List ',lcount,': ',end='')
    #     print(feat,sep=', ',end='')
    #     print()
    #     lcount += 1

    #count number of instance where face is either true or false
    faceCount = 1
    for x in range(DATA_TOTAL):
        if labels[x] == '1':
            faceCount += 1

    notFaceCount = DATA_TOTAL - faceCount
    # print(faceCount, ',', notFaceCount)

    face_data = []
    not_face_data = []

    for y in range(25):
        face_rows = []
        not_face_rows = []
        for x in range(DATA_TOTAL):
            if labels[x] == '1':
                face_rows.append(feat_list[x][y])
            else:
                not_face_rows.append(feat_list[x][y])
        face_data.append(face_rows)
        not_face_data.append(not_face_rows)


    maxList = []
    for x in face_data:
        maxList.append(max(x))

    for x in not_face_data:
        maxList.append(max(x))

    maxValue = max(maxList)

    face_data_probs = []
    for i in range(25):
        data_count = []
        for num in range(maxValue+1):
            numCount = 0
            for j in face_data[i]:
                if j == num:
                    numCount += 1
            if numCount != 0:
                data_count.append(numCount/faceCount)
            else:
                data_count.append(0.0001)

        face_data_probs.append(data_count)

    not_face_data_probs = []
    for i in range(25):
        data_count = []
        for num in range(maxValue+1):
            numCount = 0
            for j in not_face_data[i]:
                if j == num:
                    numCount += 1
            if numCount != 0:
                data_count.append(numCount/notFaceCount)
            else:
                data_count.append(0.0001)

        not_face_data_probs.append(data_count)


    c = 1
    for d in face_data_probs:
        print(c,': ',sum(d))
        c += 1

    nc = 1
    for d in not_face_data_probs:
        print(nc,': ',sum(d))
        nc += 1

    prob_face = faceCount/DATA_TOTAL
    prob_not_face = 1-prob_face

    resultList = []

    for z in range(DATA_TOTAL):
        feat_p_t = 0
        for n in range(25):
            for check in range(maxValue+1):
                if feat_list[z][n] == check:
                    feat_p_t += math.log(face_data_probs[n][check])

        feat_p_f = 0
        for n in range(25):
            for check in range(maxValue+1):
                if feat_list[z][n] == check:
                    feat_p_f += math.log(not_face_data_probs[n][check])

        face = feat_p_t*prob_face
        not_face = feat_p_f*prob_not_face

        if face > not_face:
            resultList.append(1)
            #print(z+1,': ','face = True')
        else:
            resultList.append(0)
            #print(z+1,': ','face = False')

        #print(face, ',',not_face)

    for i in range(DATA_TOTAL):
        labels[i] = int(labels[i])

    correct = 0
    for val in range(DATA_TOTAL):
        if resultList[val] == labels[val]:
            correct += 1

    accuracy = correct/DATA_TOTAL

    print('accuracy: ',round(accuracy*100,2),'%')


    # counter = 1
    # for x in labels:
    #     print(counter,": ",x)
    #     counter+=1


main()
