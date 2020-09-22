from collections import Counter
import pickle

def list2bag(ls):
    bag = []
    for l in ls:
        c = Counter()
        for element in l[:-1]:
            c[element]+=1
        bag.append(c)
    return bag
def extract_label(ls):
    label = []
    for l in ls:
        label.append(l[-1])
    return label

infile = open ('review.pickle','rb')
review = pickle.load(infile)
infile.close()
bag = list2bag(review[:1500])
outflnm = 'X_train.pickle'
outfile = open(outflnm,'wb')
pickle.dump(bag,outfile)
outfile.close()
labels = extract_label(review[:1500])
outflnm = 'y_train.pickle'
outfile = open(outflnm,'wb')
pickle.dump(labels,outfile)
outfile.close()
bag = list2bag(review[1500:])
outflnm = 'X_val.pickle'
outfile = open(outflnm,'wb')
pickle.dump(bag,outfile)
outfile.close()
labels = extract_label(review[1500:])
outflnm = 'y_val.pickle'
outfile = open(outflnm,'wb')
pickle.dump(labels,outfile)
outfile.close()
