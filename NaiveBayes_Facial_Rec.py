import math

#Read Global Files
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

#class to get 2-D list of features for every instance
class Features:
    def __init__(self, l_list, f_rows, f_cols, pix_incr_W, pix_incr_H, d_total):
        self.data_total = d_total
        self.startFeatx = 0
        self.endFeatx = pix_incr_H
        self.f_list = []
        self.f_rows = f_rows
        self.f_cols = f_cols
        self.pixel_incr_W = pix_incr_W
        self.pixel_incr_H = pix_incr_H

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

    def get_f_list(self):
        for x in range(self.data_total):
            featList = []
            self.get_features(featList)
            self.f_list.append(featList)
        return self.f_list

#count number of instances that are either true or false
def get_total_true(labels_list, d_total):
    true_count = 1
    for x in range(d_total):
        if labels_list[x] == '1':
            true_count += 1
    false_count = d_total - true_count
    return true_count, false_count

#return a tuple of 2_D lists that contain fata tables for instance label = true and label = false
def get_data_tables(f_list, labels_list, feat_total, d_total):
    true_data_table = []
    false_data_table = []
    for y in range(feat_total):
        true_rows = []
        false_rows = []
        for x in range(d_total):
            if labels_list[x] == '1':
                true_rows.append(f_list[x][y])
            else:
                false_rows.append(f_list[x][y])
        true_data_table.append(true_rows)
        false_data_table.append(false_rows)
    return true_data_table, false_data_table

#returns integer that represents the max possible value a data point can take in the data tables
def get_max_feat_value(data_tables):
    max_list = []
    for x in data_tables[0]:
        max_list.append(max(x))

    for x in data_tables[1]:
        max_list.append(max(x))

    max_f_value = max(max_list)
    return max_f_value

#return a list of computed probabilities for each data point occuring in each feature out of all instances
def get_data_feature_probs(feat_total, max_val, data_tables, tf_totals):
    data_probs = []
    for i in range(feat_total):
        data_count = []
        for num in range(max_val):
            num_count = 0
            for j in data_tables[i]:
                if j == num:
                    num_count += 1
            if num_count != 0:
                data_count.append(num_count/tf_totals)
            else:
                data_count.append(0.0001)

        data_probs.append(data_count)
    return data_probs

#returns a tuple of probabilities representing the occurance of a true instance and a false instance
def prob_instance_t_f(d_total, tf_total):
    prob_true = tf_total/d_total
    prob_false = 1-prob_true
    return prob_true, prob_false

#return a list of "1"'s and "0"'s (representing true or false) that Naive Bayes determined based on training data for each instance
def get_results(d_total, feat_total, max_val, f_list, t_data_probs, f_data_probs, prob_tf):
    results_list = []
    for z in range(d_total):
        feat_p_t = 0
        for n in range(feat_total):
            for check in range(max_val):
                if f_list[z][n] == check:
                    feat_p_t += math.log(t_data_probs[n][check])

        feat_p_f = 0
        for n in range(feat_total):
            for check in range(max_val):
                if f_list[z][n] == check:
                    feat_p_f += math.log(f_data_probs[n][check])

        true = feat_p_t*prob_tf[0]
        false = feat_p_f*prob_tf[1]

        if true > false:
            results_list.append(1)

        else:
            results_list.append(0)

    return results_list

#compare results list to actual label list and compute the accuracy of the results, then print the accuracy
def print_accuracy(labels_list, d_total, r_list):
    #convert string list to integers
    for i in range(d_total):
        labels_list[i] = int(labels_list[i])

    correct = 0
    for val in range(d_total):
        if r_list[val] == labels_list[val]:
            correct += 1

    accuracy = correct/d_total

    print('accuracy: ',round(accuracy*100,2),'%')

#main function to initialize all other functions
def main():
    features = Features(labels, FEATURE_ROWS, FEATURE_COLS, PIX_INCREMENT_W, PIX_INCREMENT_H, DATA_TOTAL)
    feat_list = features.get_f_list()
    t_f_total = get_total_true(labels, DATA_TOTAL)
    t_f_data_tables = get_data_tables(feat_list, labels, FEATURE_TOTAL, DATA_TOTAL)
    max_feature_value = get_max_feat_value(t_f_data_tables)
    true_data_feat_probs = get_data_feature_probs(FEATURE_TOTAL, max_feature_value, t_f_data_tables[0], t_f_total[0])
    false_data_feat_probs = get_data_feature_probs(FEATURE_TOTAL, max_feature_value, t_f_data_tables[1], t_f_total[1])
    prob_inst_t_f = prob_instance_t_f(DATA_TOTAL, t_f_total[0])
    result_list = get_results(DATA_TOTAL, FEATURE_TOTAL, max_feature_value, feat_list, true_data_feat_probs, false_data_feat_probs, prob_inst_t_f)
    print_accuracy(labels, DATA_TOTAL, result_list)

#init main
main()
