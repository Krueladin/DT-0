import math
import csv
import sys
import argparse

parser = argparse.ArgumentParser(description="Classification tree.")
parser.add_argument('file', metavar='F', help='Dataset csv file for classification.')
parser.add_argument('target_column', metavar='t', type=int, 
        help='Column where classification data is located.')
parser.add_argument('num_rows', nargs='?', metavar='r', type=int, default=-1, 
        help='Number of rows to read from the dataset.')


def calc_entropy(target):
    """Calculates the entropy of a single column.
    
    Entropy is the measure of uncertainty within data.
    H(S) = Sum (x in X, 1, k): -p(x)*log(p(x), k)
    Where 
        H(S) - The measure of uncertainty in a dataset 'S'.
        S - The current dataset ebing operated on.
        X - Set of classes in 'S'.
        x - A class of 'S'.
        k - The number of classes within 'S'. Also used to normalize dataset entropy to a value 
            between 0 and 1 within the log portion of the calculation.
        p(x) - The proportion of the occurance of 'x' in the dataset 'S'.
    target - An array of data representing a single column of a dataset.

    Returns the calculated entropy of the passed in column.
    """
    classes = find_label_totals(target)
    tot_classes = len(classes)
    if tot_classes <= 1:
        return 0

    # Calculate the total sum of values.
    tot_sum = 0
    for key in classes:
        tot_sum += classes[key]
    
    # Calculate the entropy for the column.
    entropy = 0.0
    for key in classes:
        percent = float(classes[key]) / tot_sum 
        entropy -= percent * math.log(percent, tot_classes)
    return entropy

def calc_info_gain(col, target):
    """Calculates the information gain according to a target column (column with desired values
    for prediction) and a column of data from the same dataset.
    
    Used for training a descision tree. 'col' must be equal in length to 'target'.
    
    Information Gain is a measure of the difference of entropy after the dataset 'S' is split
    on an attribute (column) of the dataset 'A'.
    G(A, S) = H(S) - Sum (t in T, 1, k): p(t)H(t)
    Where
        G(A, S) - Information gain of dataset 'S'.
        S - The current dataset being operated on.
        A - An attribute (data from the same label for multiple records) of the dataset 'S'.
        T - The subsets creating by splitting S by attribute A s.t. each split (new subset 't')
            has a homogenous 'a' (unique value in 'A').
        p(t) - The proportion of the number of elements in 't' to the number of elements in 'S'.
        H(t) - Entropy of subset 't'.
    col - An array of data representing a single column of a dataset.
    target - An array of data representing the information desired in classification.

    Throws an exception if 'col' and 'target' are not equal in length.
    Returns a float value representing the information gain of the parameter 'col'.
    """
    # Check that 'col' and 'target are of the same size.
    if len(col) != len(target):
        raise Exception("data are not equal lengths: {} vs. {}".format(len(col), len(target)))
    
    # Find the classes within the target column
    # and the entropy of the target column.
    t_classes = find_label_totals(target)
    t_entropy = calc_entropy(target)

    c_classes = find_label_totals(col)
    c_entropies = []
    length_arr = []
    for c_class in c_classes:
        # Grab the data associated with the target class
        # per column value.
        comb_data = [target[i] for i in range(len(col)) if col[i] == c_class]
        if len(comb_data) <= 0:
            c_entropies.append(0.0)
            continue
        # Calculate the weight, p(t) for this separation.
        weight = float(len(comb_data)) / len(col)
        # Add the entropy of the column class to the entropies array.
        c_entropies.append(weight * calc_entropy(comb_data))
    
    # Calculate the average entropy of the column.
    #   The sum of each p(t)H(t).
    avg_entropy = 0.0
    for item in c_entropies:
        avg_entropy += item
    # Return the information gain.
    return t_entropy - avg_entropy

def find_label_totals(col):
    """Takes a data column and finds each unique value.

    col - An array of data representing a single column of a dataset.

    Returns a map with key for each unique element in the array 'col' passed in as a parameter.
        The values for this map indicate the frequency of appearance of each unique value.
    """
    val_dict = {}
    for k in col:
        if k not in val_dict:
            val_dict[k] = 0 
        val_dict[k] += 1
    return val_dict 

def read_data_file(file, target_col, lines_to_read=-1):
    """Opens and reads a csv data file and returns a dataset, column major, and a target set
    cooresponding to the associated rows.
    
    file - The file to read.
    target_col - The column where classifier data is located.
    lines_to_read - The number of rows of the dataset to read. If # of lines specified is greater
                    than what is located in the file, function returns as many rows located 
                    in the file.
    Returns a pair, an array organized column-major and the cooresponding target column. 
        e.g. (cols, target) = read_data_file('dataset.csv', 0, 100)
            # Reads the first 100 rows of a dataset 'dataset.csv' with 
            # the classifier objective row of '0'.
            cols[0] # Returns the first column of the dataset that is not part of classification.
    """
    cols = []
    # Put together the first number of specified lines of the dataset.
    with open(file, 'rb') as csvfile:
        if lines_to_read < 1:
            lines_to_read = sum(1 for row in csvfile)
            csvfile.seek(0)
    
        reader = csv.reader(csvfile, delimiter=",", quotechar='|')
        for i, row in zip(range(lines_to_read), reader):
            for j, col in zip(range(len(row)), row):
                if len(cols) <= j:
                    cols.append([])
                cols[j].append(col)
            if i == lines_to_read:
                break
    # Calculate the information gain of each of the columns.
    # Select the column giving the most information gain.
    target = cols[target_col]
    del cols[target_col]
    return (cols, target)

def main():
    print "Started"
    args = parser.parse_args()
    (cols, target) = read_data_file(args.file, args.target_column, args.num_rows)
    col_gain = []
    for col in cols:
        col_gain.append(calc_info_gain(col, target))

    print col_gain

if __name__ == "__main__":
    main()
