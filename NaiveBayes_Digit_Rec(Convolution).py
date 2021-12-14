import math
import numpy as np
import random
#Read Global Files
#read training data into list
faceFile = open(r"C:\Users\atfan\github\AI_Final_Project\digitdata\trainingimages",'r')
train_lines = faceFile.readlines()
faceFile.close()
#read training data labels into a list
labelFile = open(r"C:\Users\atfan\github\AI_Final_Project\digitdata\traininglabels",'r')
train_labels = []
for c in labelFile.read():
    if c != '\n' and c != ' ':
        train_labels.append(c)
labelFile.close()
#read testing data into list
faceFile = open(r"C:\Users\atfan\github\AI_Final_Project\digitdata\testimages",'r')
test_lines = faceFile.readlines()
faceFile.close()
#read testing data labels into a list
labelFile = open(r"C:\Users\atfan\github\AI_Final_Project\digitdata\testlabels",'r')
test_labels = []
for c in labelFile.read():
    if c != '\n' and c != ' ':
        test_labels.append(c)
labelFile.close()

#Global Variables
ITERATIONS = 5
DATA_PERCENT = 1
SAMPLE_TOTAL = len(train_labels)
TEST_DATA_TOTAL = len(test_labels)
IMAGE_PIX_WIDTH = 28
IMAGE_PIX_HEIGHT = 28
# TRAIN_FILE_PIX_HEIGHT = TRAIN_DATA_TOTAL*IMAGE_PIX_HEIGHT
# TEST_FILE_PIX_HEIGHT = TEST_DATA_TOTAL*IMAGE_PIX_HEIGHT

FILTER_DIM = 3
filter = [[0, 1, 0],
          [0, 1, 0],
          [0, 1, 0]]

def convert_image_to_bin(lines, file_pix_H, img_pix_H):
    new_image_list = []
    end_incr = img_pix_H
    for x in range(0, file_pix_H, img_pix_H):
        new_lines_list = []
        for line in lines[x:end_incr]:
            new_line_list = []
            for char in line[:28]:
                if char == '#' or char == '+':
                    new_line_list.append(1)
                else:
                    new_line_list.append(0)

            new_lines_list.append(new_line_list)

        new_image_list.append(new_lines_list)
        end_incr += img_pix_H

    return new_image_list

#class to get 2-D list of features for every instance
class Features:
    def __init__(self, filtr, filter_dim, d_total, data):
        self.data_total = d_total
        self.all_img_f_list = []
        self.data = data
        self.filter = filtr
        self.filter_dim = filter_dim
        self.filtered_matrix = []

    def get_features(self, index):
        #initiate convolution
        self.data[index] = np.asarray(self.data[index])
        filter = np.asarray(self.filter)
        sub_shape = (self.filter_dim, self.filter_dim)
        view_shape = tuple(np.subtract(self.data[index].shape, sub_shape) + 1) + sub_shape
        strides = self.data[index].strides + self.data[index].strides
        sub_matrices = np.lib.stride_tricks.as_strided(self.data[index],view_shape,strides)

        row_count = 0
        for x in self.data[index]:
            col_count = 0
            for y in x:
                col_count += 1
            row_count += 1

        filtered_matrix = []
        for a in range(row_count-3):
            dot_prod_list = []
            for b in range(col_count-3):
                sum = 0
                for x in range(self.filter_dim):
                    for y in range(self.filter_dim):
                        sum += self.filter[x][y]*sub_matrices[a][b][x][y]
                dot_prod_list.append(sum)
            filtered_matrix.append(dot_prod_list)

        #initiate feature pooling
        filtered_matrix = np.asarray(filtered_matrix)
        sub_shape = (2, 2)
        view_shape = tuple(np.subtract(filtered_matrix.shape, sub_shape) + 1) + sub_shape
        strides = filtered_matrix.strides + filtered_matrix.strides
        filtered_sub_matrices = np.lib.stride_tricks.as_strided(filtered_matrix,view_shape,strides)

        f_row_count = 0
        for x in filtered_matrix:
            f_col_count = 0
            for y in x:
                f_col_count += 1
            f_row_count += 1

        #max pool and flatten features to 1D list
        pooled_matrix = []
        for a in range(f_row_count-2):
            pooled_cols = []
            for b in range(f_col_count-2):
                pooled_cols.append(np.amax(filtered_sub_matrices[a][b]))
            pooled_matrix.append(pooled_cols)

        f_list = []
        for rows in pooled_matrix:
            for char in rows:
                f_list.append(char)

        feature_total = len(f_list)

        return f_list, feature_total


    #returns a 2D list of all feature values across all instances in a data file
    def get_feat_all_img(self):
        for x in range(self.data_total):
            img_features = self.get_features(x)
            self.all_img_f_list.append(img_features[0])
        feature_total = img_features[1]
        return self.all_img_f_list, feature_total

#count number of instances that are either true or false
def get_digit_totals(labels_list, d_total):
    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0
    count_5 = 0
    count_6 = 0
    count_7 = 0
    count_8 = 0
    count_9 = 0
    for x in range(d_total):
        if labels_list[x] == '0':
            count_0 += 1
        elif labels_list[x] == '1':
            count_1 += 1
        elif labels_list[x] == '2':
            count_2 += 1
        elif labels_list[x] == '3':
            count_3 += 1
        elif labels_list[x] == '4':
            count_4 += 1
        elif labels_list[x] == '5':
            count_5 += 1
        elif labels_list[x] == '6':
            count_6 += 1
        elif labels_list[x] == '7':
            count_7 += 1
        elif labels_list[x] == '8':
            count_8 += 1
        elif labels_list[x] == '9':
            count_9 += 1
        else:
            continue
    return count_0, count_1, count_2, count_3, count_4, count_5, count_6, count_7, count_8, count_9

#return a tuple of 2_D lists that contain data tables for instance label = true and label = false
def get_data_tables(f_list, labels_list, feat_total, d_total):
    data_table_0 = []
    data_table_1 = []
    data_table_2 = []
    data_table_3 = []
    data_table_4 = []
    data_table_5 = []
    data_table_6 = []
    data_table_7 = []
    data_table_8 = []
    data_table_9 = []
    for y in range(feat_total):
        data_rows_0 = []
        data_rows_1 = []
        data_rows_2 = []
        data_rows_3 = []
        data_rows_4 = []
        data_rows_5 = []
        data_rows_6 = []
        data_rows_7 = []
        data_rows_8 = []
        data_rows_9 = []
        for x in range(d_total):
            if labels_list[x] == '0':
                data_rows_0.append(f_list[x][y])
            elif labels_list[x] == '1':
                data_rows_1.append(f_list[x][y])
            elif labels_list[x] == '2':
                data_rows_2.append(f_list[x][y])
            elif labels_list[x] == '3':
                data_rows_3.append(f_list[x][y])
            elif labels_list[x] == '4':
                data_rows_4.append(f_list[x][y])
            elif labels_list[x] == '5':
                data_rows_5.append(f_list[x][y])
            elif labels_list[x] == '6':
                data_rows_6.append(f_list[x][y])
            elif labels_list[x] == '7':
                data_rows_7.append(f_list[x][y])
            elif labels_list[x] == '8':
                data_rows_8.append(f_list[x][y])
            elif labels_list[x] == '9':
                data_rows_9.append(f_list[x][y])
            else:
                continue

        data_table_0.append(data_rows_0)
        data_table_1.append(data_rows_1)
        data_table_2.append(data_rows_2)
        data_table_3.append(data_rows_3)
        data_table_4.append(data_rows_4)
        data_table_5.append(data_rows_5)
        data_table_6.append(data_rows_6)
        data_table_7.append(data_rows_7)
        data_table_8.append(data_rows_8)
        data_table_9.append(data_rows_9)

    return data_table_0, data_table_1, data_table_2, data_table_3, data_table_4, data_table_5, data_table_6, data_table_7, data_table_8, data_table_9

#returns integer that represents the max possible value a data point can take in the data tables
def get_max_feat_value(data_tables):
    max_list = []
    for y in range(10):
        for x in data_tables[y]:
            max_list.append(max(x))

    max_f_value = max(max_list)+1
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
def prob_instance_prior(d_total, dig_total):
    prob_0 = dig_total[0]/d_total
    prob_1 = dig_total[1]/d_total
    prob_2 = dig_total[2]/d_total
    prob_3 = dig_total[3]/d_total
    prob_4 = dig_total[4]/d_total
    prob_5 = dig_total[5]/d_total
    prob_6 = dig_total[6]/d_total
    prob_7 = dig_total[7]/d_total
    prob_8 = dig_total[8]/d_total
    prob_9 = dig_total[9]/d_total
    return prob_0, prob_1, prob_2, prob_3, prob_4, prob_5, prob_6, prob_7, prob_8, prob_9

#return a list of "1"'s and "0"'s (representing true or false) that Naive Bayes determined based on training data for each instance
def get_results(d_total, feat_total, max_val, f_list, data_probs, prob_prior):
    results_list = []
    for z in range(d_total):
        probs_list = []
        for dig in range(10):
            feat_prob_sum = 0
            for n in range(feat_total):
                for check in range(max_val):
                    if f_list[z][n] == check:
                        feat_prob_sum += math.log(data_probs[dig][n][check])
            probs_list.append(feat_prob_sum*prob_prior[dig])

        check_max = probs_list[0]
        max_index = 0
        for x in range(1,10):
            if probs_list[x] > check_max:
                check_max = probs_list[x]
                max_index = x
        results_list.append(max_index)

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

    return accuracy

#main function to initialize all other functions
def main(percent):
    #set sample size based on a percentage of total data set
    TRAIN_DATA_TOTAL = int(round(len(train_labels)*percent, 0))
    TRAIN_FILE_PIX_HEIGHT = TRAIN_DATA_TOTAL*IMAGE_PIX_HEIGHT
    TEST_FILE_PIX_HEIGHT = TEST_DATA_TOTAL*IMAGE_PIX_HEIGHT
    # print(TRAIN_DATA_TOTAL)
    #match labels with images in a list
    start = 0
    end = 28
    train_img_labels = []
    for x in range(SAMPLE_TOTAL):
        image = []
        lab_img = []
        for line in train_lines[start:end]:
                image.append(line)
        lab_img.append(image)
        lab_img.append(train_labels[x])
        train_img_labels.append(lab_img)
        start+=28
        end+=28
    #randomize data based on sample size
    rand_train_img_label = random.sample(train_img_labels, TRAIN_DATA_TOTAL)
    #seperate images and labels into distinct lists
    new_train_img = []
    new_train_labels = []
    for i in rand_train_img_label:
        for j in i[0]:
            new_train_img.append(j)
        new_train_labels.append(i[1])
    #Initialize classes to get feature lists from training data
    train_bin_data_list = convert_image_to_bin(new_train_img, TRAIN_FILE_PIX_HEIGHT, IMAGE_PIX_HEIGHT)
    train_features = Features(filter, FILTER_DIM, TRAIN_DATA_TOTAL, train_bin_data_list)
    train_feat_list = train_features.get_feat_all_img()
    #Initialize classes to get feature lists from testing data
    test_bin_data_list = convert_image_to_bin(test_lines, TEST_FILE_PIX_HEIGHT, IMAGE_PIX_HEIGHT)
    test_features = Features(filter, FILTER_DIM, TEST_DATA_TOTAL, test_bin_data_list)
    test_feat_list = test_features.get_feat_all_img()
    #Initialize training functions
    digit_totals = get_digit_totals(new_train_labels, TRAIN_DATA_TOTAL)
    digit_data_tables = get_data_tables(train_feat_list[0], new_train_labels, train_feat_list[1], TRAIN_DATA_TOTAL)
    max_feature_value = get_max_feat_value(digit_data_tables)
    digit_probs_list = []
    for p in range(10):
        digit_probs_list.append(get_data_feature_probs(train_feat_list[1], max_feature_value, digit_data_tables[p], digit_totals[p]))
    prob_inst_prior = prob_instance_prior(TRAIN_DATA_TOTAL, digit_totals)
    #Initialize testing functions
    result_list = get_results(TEST_DATA_TOTAL, test_feat_list[1], max_feature_value, test_feat_list[0], digit_probs_list, prob_inst_prior)
    acc = print_accuracy(test_labels, TEST_DATA_TOTAL, result_list)

    return round(acc*100,2)

#init main
acc_list = []
for x in range(ITERATIONS):
    acc_list.append(main(DATA_PERCENT))
mean_acc = sum(acc_list)/len(acc_list)
var_mean = 0
for i in acc_list:
    var_mean += pow((i-mean_acc),2)
var = var_mean/len(acc_list)
std_dev = math.sqrt(var)
print('mean accuracy: ',round(mean_acc,2),'%')
print('variance: ', round(var,2))
print('standard deviation: ',round(std_dev,3))
