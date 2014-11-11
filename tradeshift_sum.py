import pandas as pd
import numpy as np
import math
import time
import collections

start = time.time()

dir = ''
pred_list = ['quick_start_all.csv','submission#14_4.csv','submission_h2o.csv']

n=len(pred_list)
doc = pd.DataFrame(columns=['id_label','pred'])

d = {}

n = n + 1

for i in pred_list:
	print i
	koef = 2.0/n if i == 'submission#14_4.csv' else 1.0/n
	doc1 = pd.read_csv(dir + i, skiprows = 1, names=['id_label','pred'])
	for index, row in doc1.iterrows():
		if row['id_label'] in d:
			d[row['id_label']] += (float(row['pred']) + 10**(-14)) * koef
		else:	
			d[row['id_label']] = (float(row['pred']) + 10**(-14)) * koef

print "Almost done..."

doc = pd.DataFrame(d.items(), columns=['id_label', 'pred'])

import gzip

def save_predictions(name, ids, predictions) :
    out = gzip.open(name, 'w')
    print >>out, 'id_label,pred'
    for index, row in doc.iterrows():
        print >>out, row['id_label'] + ',' + str(row['pred'])

save_predictions('tradeshift_sum#10.csv.gz', doc['id_label'], doc['pred'])

#weighted submission gets around 0.0061 on private leaderboard, enough for postition 71/395 (top 25%)

