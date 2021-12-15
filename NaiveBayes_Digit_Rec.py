import math
import random

# Read Global Files
# read training data into list
faceFile = open(r"C:\Users\atfan\github\AI_Final_Project\digitdata\trainingimages", "r")
train_lines = faceFile.readlines()
faceFile.close()
# read training data labels into a list
labelFile = open(
    r"C:\Users\atfan\github\AI_Final_Project\digitdata\traininglabels", "r"
)
train_labels = []
for c in labelFile.read():
    if c != "\n" and c != " ":
        train_labels.append(c)
labelFile.close()
# read testing data into list
faceFile = open(r"C:\Users\atfan\github\AI_Final_Project\digitdata\testimages", "r")
test_lines = faceFile.readlines()
faceFile.close()
# read testing data labels into a list
labelFile = open(r"C:\Users\atfan\github\AI_Final_Project\digitdata\testlabels", "r")
test_labels = []
for c in labelFile.read():
    if c != "\n" and c != " ":
        test_labels.append(c)
labelFile.close()

# Global Variables
ITERATIONS = 5
DATA_PERCENT = 1
SAMPLE_TOTAL = len(train_labels)
TEST_DATA_TOTAL = len(test_labels)
FEATURE_ROWS = 28
FEATURE_COLS = 28
FEATURE_TOTAL = FEATURE_ROWS * FEATURE_COLS
IMAGE_PIX_WIDTH = 28
IMAGE_PIX_HEIGHT = 28
PIX_INCREMENT_W = IMAGE_PIX_WIDTH // FEATURE_ROWS
PIX_INCREMENT_H = IMAGE_PIX_HEIGHT // FEATURE_COLS

# class to get 2-D list of features for every instance
class Features:
    def __init__(self, f_rows, f_cols, pix_incr_W, pix_incr_H, d_total, lines):
        self.data_total = d_total
        self.startFeatx = 0
        self.endFeatx = pix_incr_H
        self.f_list = []
        self.f_rows = f_rows
        self.f_cols = f_cols
        self.pixel_incr_W = pix_incr_W
        self.pixel_incr_H = pix_incr_H
        self.lines = lines

    def get_features(self, f_list):
        numFeat = 1
        for rows in range(self.f_rows):
            startFeaty = 0
            endFeaty = self.pixel_incr_W
            for cols in range(self.f_cols):
                count = 0
                for line in self.lines[self.startFeatx : self.endFeatx]:
                    startFeaty = startFeaty
                    endFeaty = endFeaty
                    for char in line[startFeaty:endFeaty]:
                        if char == "#" or char == "+":
                            count += 1
                f_list.append(count)
                startFeaty += self.pixel_incr_W
                endFeaty += self.pixel_incr_W
                numFeat += 1
            self.startFeatx += self.pixel_incr_H
            self.endFeatx += self.pixel_incr_H

    # returns a 2D list of all feature values across all instances in a data file
    def get_f_list(self):
        for x in range(self.data_total):
            featList = []
            self.get_features(featList)
            self.f_list.append(featList)
        return self.f_list


# count number of instances that are either true or false
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
        if labels_list[x] == "0":
            count_0 += 1
        elif labels_list[x] == "1":
            count_1 += 1
        elif labels_list[x] == "2":
            count_2 += 1
        elif labels_list[x] == "3":
            count_3 += 1
        elif labels_list[x] == "4":
            count_4 += 1
        elif labels_list[x] == "5":
            count_5 += 1
        elif labels_list[x] == "6":
            count_6 += 1
        elif labels_list[x] == "7":
            count_7 += 1
        elif labels_list[x] == "8":
            count_8 += 1
        elif labels_list[x] == "9":
            count_9 += 1
        else:
            continue

    return (
        count_0,
        count_1,
        count_2,
        count_3,
        count_4,
        count_5,
        count_6,
        count_7,
        count_8,
        count_9,
    )


# return a tuple of 2_D lists that contain data tables for instance label = true and label = false
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
            if labels_list[x] == "0":
                data_rows_0.append(f_list[x][y])
            elif labels_list[x] == "1":
                data_rows_1.append(f_list[x][y])
            elif labels_list[x] == "2":
                data_rows_2.append(f_list[x][y])
            elif labels_list[x] == "3":
                data_rows_3.append(f_list[x][y])
            elif labels_list[x] == "4":
                data_rows_4.append(f_list[x][y])
            elif labels_list[x] == "5":
                data_rows_5.append(f_list[x][y])
            elif labels_list[x] == "6":
                data_rows_6.append(f_list[x][y])
            elif labels_list[x] == "7":
                data_rows_7.append(f_list[x][y])
            elif labels_list[x] == "8":
                data_rows_8.append(f_list[x][y])
            elif labels_list[x] == "9":
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

    return (
        data_table_0,
        data_table_1,
        data_table_2,
        data_table_3,
        data_table_4,
        data_table_5,
        data_table_6,
        data_table_7,
        data_table_8,
        data_table_9,
    )


# returns integer that represents the max possible value a data point can take in the data tables
def get_max_feat_value(data_tables):
    max_list = []
    for y in range(10):
        for x in data_tables[y]:
            max_list.append(max(x))

    max_f_value = max(max_list)
    return max_f_value


# return a list of computed probabilities for each data point occuring in each feature out of all instances
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
                data_count.append(num_count / tf_totals)
            else:
                data_count.append(0.0001)

        data_probs.append(data_count)
    return data_probs


# returns a tuple of probabilities representing the occurance of a true instance and a false instance
def prob_instance_prior(d_total, dig_total):
    prob_0 = dig_total[0] / d_total
    prob_1 = dig_total[1] / d_total
    prob_2 = dig_total[2] / d_total
    prob_3 = dig_total[3] / d_total
    prob_4 = dig_total[4] / d_total
    prob_5 = dig_total[5] / d_total
    prob_6 = dig_total[6] / d_total
    prob_7 = dig_total[7] / d_total
    prob_8 = dig_total[8] / d_total
    prob_9 = dig_total[9] / d_total
    return (
        prob_0,
        prob_1,
        prob_2,
        prob_3,
        prob_4,
        prob_5,
        prob_6,
        prob_7,
        prob_8,
        prob_9,
    )


# return a list of "1"'s and "0"'s (representing true or false) that Naive Bayes determined based on training data for each instance
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
            probs_list.append(feat_prob_sum * prob_prior[dig])

        check_max = probs_list[0]
        max_index = 0
        for x in range(1, 10):
            if probs_list[x] > check_max:
                check_max = probs_list[x]
                max_index = x
        results_list.append(max_index)

    return results_list


# compare results list to actual label list and compute the accuracy of the results, then print the accuracy
def print_accuracy(labels_list, d_total, r_list):
    # convert string list to integers
    for i in range(d_total):
        labels_list[i] = int(labels_list[i])

    correct = 0
    for val in range(d_total):
        if r_list[val] == labels_list[val]:
            correct += 1

    accuracy = correct / d_total

    print("accuracy: ", round(accuracy * 100, 2), "%")

    return accuracy


# main function to initialize all other functions
def main(percent):
    # set sample size based on a percentage of total data set
    TRAIN_DATA_TOTAL = int(round(len(train_labels) * percent, 0))
    # print(TRAIN_DATA_TOTAL)
    # match labels with images in a list
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
        start += 28
        end += 28
    # randomize data based on sample size
    rand_train_img_label = random.sample(train_img_labels, TRAIN_DATA_TOTAL)
    # seperate images and labels into distinct lists
    new_train_img = []
    new_train_labels = []
    for i in rand_train_img_label:
        for j in i[0]:
            new_train_img.append(j)
        new_train_labels.append(i[1])
    # Initialize classes to get feature lists from training and testing
    features = Features(
        FEATURE_ROWS,
        FEATURE_COLS,
        PIX_INCREMENT_W,
        PIX_INCREMENT_H,
        TRAIN_DATA_TOTAL,
        new_train_img,
    )
    train_feat_list = features.get_f_list()
    features = Features(
        FEATURE_ROWS,
        FEATURE_COLS,
        PIX_INCREMENT_W,
        PIX_INCREMENT_H,
        TEST_DATA_TOTAL,
        test_lines,
    )
    test_feat_list = features.get_f_list()
    # Initialize training functions
    digit_totals = get_digit_totals(new_train_labels, TRAIN_DATA_TOTAL)
    digit_data_tables = get_data_tables(
        train_feat_list, new_train_labels, FEATURE_TOTAL, TRAIN_DATA_TOTAL
    )
    max_feature_value = get_max_feat_value(digit_data_tables)
    digit_probs_list = []
    for p in range(10):
        digit_probs_list.append(
            get_data_feature_probs(
                FEATURE_TOTAL,
                max_feature_value + 1,
                digit_data_tables[p],
                digit_totals[p],
            )
        )
    prob_inst_prior = prob_instance_prior(TRAIN_DATA_TOTAL, digit_totals)
    # Initialize testing functions
    result_list = get_results(
        TEST_DATA_TOTAL,
        FEATURE_TOTAL,
        max_feature_value + 1,
        test_feat_list,
        digit_probs_list,
        prob_inst_prior,
    )
    acc = print_accuracy(test_labels, TEST_DATA_TOTAL, result_list)

    return round(acc * 100, 2)


# init main
acc_list = []
for x in range(ITERATIONS):
    acc_list.append(main(DATA_PERCENT))
mean_acc = sum(acc_list) / len(acc_list)
var_mean = 0
for i in acc_list:
    var_mean += pow((i - mean_acc), 2)
var = var_mean / len(acc_list)
std_dev = math.sqrt(var)
print("mean accuracy: ", round(mean_acc, 2), "%")
print("variance: ", round(var, 2))
print("standard deviation: ", round(std_dev, 3))
