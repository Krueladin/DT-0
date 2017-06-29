import math
import csv

def calc_entropy(target):
    classes = find_label_totals(target)
    tot_classes = len(classes)
    if tot_classes <= 1:
        return 0

#    print classes
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
        # Calculate the weight for this separation.
        weight = float(len(comb_data)) / len(col)
        # Add the entropy of the column class to the entropies array.
        c_entropies.append(weight * calc_entropy(comb_data))
    
    # Calculate the average entropy of the column.
    avg_entropy = 0.0
    for item in c_entropies:
        avg_entropy += item
    # Return the information gain.
    return t_entropy - avg_entropy

def find_label_totals(col):
    val_dict = {}
    for k in col:
        if k not in val_dict:
            val_dict[k] = 0 
        val_dict[k] += 1
    return val_dict 

def main():
    print "Started"
    cols = []
    
    # Put together the first 100 lines of the dataset.
    with open('agaricus-lepiota.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='|')
        for i, row in zip(range(100), reader):
            for j, col in zip(range(len(row)), row):
                if len(cols) <= j:
                    cols.append([])
                cols[j].append(col)
            if i == 100:
                break
    
    # Calculate the information gain of each of the columns.
    # Select the column giving the most information gain.
    target = cols[0]
    del cols[0]
#    print target
#    print cols
    col_gain = []
    for col in cols:
        col_gain.append(calc_info_gain(col, target))

    print col_gain

if __name__ == "__main__":
    main()
