from datetime import datetime
from math import log, exp, sqrt

# parameters #################################################################

train = 'train.csv'  # path to training file
label = 'trainLabels.csv'  # path to label file of training data
test = 'test.csv'  # path to testing file

solution = 14

monitor = open('diag'+str(solution)+'.out','w')

D = 2 ** 24  # number of weights use for each model, we have 32 of them
alpha = .1   # learning rate for sgd optimization

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# function, generator definitions ############################################

# A. x, y generator
# INPUT:
#     path: path to train.csv or test.csv
#     label_path: (optional) path to trainLabels.csv
# YIELDS:
#     ID: id of the instance (can also acts as instance count)
#     x: a list of indices that its value is 1
#     y: (if label_path is present) label value of y1 to y33
def data(path, label_path=None):
    for t, line in enumerate(open(path)):
        # initialize our generator
        if t == 0:
            # create a static x,
            # so we don't have to construct a new x for every instance
            x = [0] * (146 + 46 + 1 + 7 + 2 + 7 + 10 + 7 + 7 + 7 + 4 + 5 + 14 + 60 + 20 + 5 + 60 + 20 + 5)
            if label_path:
                label = open(label_path)
                label.readline()  # we don't need the headers
            continue
        # parse x
        row = line.rstrip().split(',')
        for m, feat in enumerate(line.rstrip().split(',')):
            if m == 0:
                ID = int(feat)
            else:
                # one-hot encode everything with hash trick
                # categorical: one-hotted
                # boolean: ONE-HOTTED
                # numerical: ONE-HOTTED!
                # note, the build in hash(), although fast is not stable,
                #       i.e., same value won't always have the same hash
                #       on different machines
                if is_number(feat):
                    feat=str(round(float(feat),1))
                x[m] = abs(hash(str(m) + '_' + feat)) % D

        hash_cols = [3,4,34,35,61,64,65,91,94,95]
        t = 145
        for i in xrange(10):
            for j in xrange(i+1,10):
                t += 1
                x[t] = abs(hash(row[hash_cols[i]]+"_x_"+row[hash_cols[j]])) % D

        #+1
        nul = 0 if row[1] == "" else 1
        x[192] = abs(hash(str(nul) + '_nul')) % D

        #+7
        hesh = 'a' if float(row[5]) == 0 else ('b' if float(row[5]) >= 1 else 'c')
        x[193]= abs(hash('i5_' + hesh)) % D
        hesh = 'a' if float(row[6]) == 0 else ('b' if float(row[6]) >= 1 else 'c')
        x[194]= abs(hash('i6_' + hesh)) % D
        hesh = 'a' if float(row[7]) == 0 else ('b' if float(row[7]) == 1 else 'c')
        x[195]= abs(hash('i7_' + hesh)) % D
        hesh = 'a' if float(row[8]) == 0 else ('b' if float(row[8]) >= 1 else 'c')
        x[196]= abs(hash('i8_' + hesh)) % D
        hesh = 'a' if float(row[9]) == 0 else ('b' if float(row[9]) >= 1 else 'c')
        x[197]= abs(hash('i9_' + hesh)) % D
        hesh = 0 if float(row[5]) == 0 and float(row[6]) == 0 and float(row[7]) == 0 and float(row[8]) == 0 and float(row[9]) else 1
        x[198]= abs(hash('i56789_' + str(hesh))) % D
        hesh = round(float(row[5]) + float(row[6]) + float(row[7]) + float(row[8]) + float(row[9]),2)
        x[199]= abs(hash('i56789_sum' + str(round(float(hesh),2)))) % D

        #+2
        yesno_cols = [1,2,10,11,12,13,14,24,25,26,30,31,32,33,41,42,43,44,45,55,56,57,62,63,71,72,73,74,75,85,86,87,92,93,101,102,103,104,105,115,116,117,126,127,128,129,130,140,142]
        yesno_hash = ""
        for i in xrange(len(yesno_cols)):
            yesno_hash += row[yesno_cols[i]] + "_x_"
        x[200] = abs(hash(yesno_hash)) % D

        nul = 0 if row[5] == 0 else 1
        x[201] = abs(hash(str(nul) + '5_nul')) % D

        #+7
        hesh = 'a' if float(row[36]) == 0 else ('b' if float(row[36]) >= 1 else 'c')
        x[202]= abs(hash('i36_' + hesh)) % D
        hesh = 'a' if float(row[37]) == 0 else ('b' if float(row[37]) >= 1 else 'c')
        x[203]= abs(hash('i37_' + hesh)) % D
        hesh = 'a' if float(row[38]) == 0 else ('b' if float(row[38]) == 1 else 'c')
        x[204]= abs(hash('i38_' + hesh)) % D
        hesh = 'a' if float(row[39]) == 0 else ('b' if float(row[39]) >= 1 else 'c')
        x[205]= abs(hash('i39_' + hesh)) % D
        hesh = 'a' if float(row[40]) == 0 else ('b' if float(row[40]) >= 1 else 'c')
        x[206]= abs(hash('i40_' + hesh)) % D
        hesh = 0 if float(row[36]) == 0 and float(row[37]) == 0 and float(row[38]) == 0 and float(row[39]) == 0 and float(row[40]) else 1
        x[207]= abs(hash('i3637383940_' + str(hesh))) % D
        hesh = round(float(row[36]) + float(row[37]) + float(row[38]) + float(row[39]) + float(row[40]),2)
        x[208]= abs(hash('i3637383940__sum' + str(round(float(hesh),2)))) % D

        #+10
        hesh = 'a' if float(row[20]) == 0 else ('b' if float(row[20]) > 0 else 'c')
        x[209]= abs(hash('i20_' + hesh)) % D
        nul = 1 if float(row[21]) == 1 else 0
        x[210] = abs(hash(str(nul) + '21_nul')) % D

        hesh = 'a' if float(row[51]) == 0 else ('b' if float(row[51]) > 0 else 'c')
        x[211]= abs(hash('i51_' + hesh)) % D
        nul = 1 if float(row[52]) == 1 else 0
        x[212] = abs(hash(str(nul) + '52_nul')) % D

        hesh = 'a' if float(row[81]) == 0 else ('b' if float(row[81]) > 0 else 'c')
        x[213]= abs(hash('i81_' + hesh)) % D
        nul = 1 if float(row[82]) == 1 else 0
        x[214] = abs(hash(str(nul) + '82_nul')) % D

        hesh = 'a' if float(row[111]) == 0 else ('b' if float(row[111]) > 0 else 'c')
        x[215]= abs(hash('i111_' + hesh)) % D
        nul = 1 if float(row[112]) == 1 else 0
        x[216] = abs(hash(str(nul) + '112_nul')) % D

        hesh = 'a' if float(row[136]) == 0 else ('b' if float(row[136]) > 0 else 'c')
        x[217]= abs(hash('i136_' + hesh)) % D
        nul = 1 if float(row[137]) == 1 else 0
        x[218] = abs(hash(str(nul) + '137_nul')) % D

        #+7
        hesh = 'a' if float(row[66]) == 0 else ('b' if float(row[66]) >= 1 else 'c')
        x[219]= abs(hash('i66_' + hesh)) % D
        hesh = 'a' if float(row[67]) == 0 else ('b' if float(row[67]) >= 1 else 'c')
        x[220]= abs(hash('i67_' + hesh)) % D
        hesh = 'a' if float(row[68]) == 0 else ('b' if float(row[68]) == 1 else 'c')
        x[221]= abs(hash('i68_' + hesh)) % D
        hesh = 'a' if float(row[69]) == 0 else ('b' if float(row[69]) >= 1 else 'c')
        x[222]= abs(hash('i69_' + hesh)) % D
        hesh = 'a' if float(row[70]) == 0 else ('b' if float(row[70]) >= 1 else 'c')
        x[223]= abs(hash('i70_' + hesh)) % D
        hesh = 0 if float(row[66]) == 0 and float(row[67]) == 0 and float(row[68]) == 0 and float(row[69]) == 0 and float(row[70]) else 1
        x[224]= abs(hash('i6667686970_' + str(hesh))) % D
        hesh = round(float(row[66]) + float(row[67]) + float(row[68]) + float(row[69]) + float(row[70]),2)
        x[225]= abs(hash('i6667686970__sum' + str(round(float(hesh),2)))) % D
        
        #+7
        hesh = 'a' if float(row[96]) == 0 else ('b' if float(row[96]) >= 1 else 'c')
        x[226]= abs(hash('i96_' + hesh)) % D
        hesh = 'a' if float(row[97]) == 0 else ('b' if float(row[97]) >= 1 else 'c')
        x[227]= abs(hash('i97_' + hesh)) % D
        hesh = 'a' if float(row[98]) == 0 else ('b' if float(row[98]) == 1 else 'c')
        x[228]= abs(hash('i98_' + hesh)) % D
        hesh = 'a' if float(row[99]) == 0 else ('b' if float(row[99]) >= 1 else 'c')
        x[229]= abs(hash('i99_' + hesh)) % D
        hesh = 'a' if float(row[100]) == 0 else ('b' if float(row[100]) >= 1 else 'c')
        x[230]= abs(hash('i100_' + hesh)) % D
        hesh = 0 if float(row[96]) == 0 and float(row[97]) == 0 and float(row[98]) == 0 and float(row[99]) == 0 and float(row[100]) else 1
        x[231]= abs(hash('i96979899100_' + str(hesh))) % D
        hesh = round(float(row[96]) + float(row[97]) + float(row[98]) + float(row[99]) + float(row[100]),2)
        x[232]= abs(hash('i96979899100__sum' + str(round(float(hesh),2)))) % D
          
        #+7
        hesh = 'a' if float(row[121]) == 0 else ('b' if float(row[121]) >= 1 else 'c')
        x[233]= abs(hash('i121_' + hesh)) % D
        hesh = 'a' if float(row[122]) == 0 else ('b' if float(row[122]) >= 1 else 'c')
        x[234]= abs(hash('i122_' + hesh)) % D
        hesh = 'a' if float(row[123]) == 0 else ('b' if float(row[123]) == 1 else 'c')
        x[235]= abs(hash('i123_' + hesh)) % D
        hesh = 'a' if float(row[124]) == 0 else ('b' if float(row[124]) >= 1 else 'c')
        x[236]= abs(hash('i124_' + hesh)) % D
        hesh = 'a' if float(row[124]) == 0 else ('b' if float(row[124]) >= 1 else 'c')
        x[237]= abs(hash('i124_' + hesh)) % D
        hesh = 0 if float(row[121]) == 0 and float(row[122]) == 0 and float(row[123]) == 0 and float(row[124]) == 0 and float(row[125]) else 1
        x[238]= abs(hash('i121122123124125_' + str(hesh))) % D
        hesh = round(float(row[121]) + float(row[122]) + float(row[123]) + float(row[124]) + float(row[125]),2)
        x[239]= abs(hash('121122123124125__sum' + str(round(float(hesh),2)))) % D

        #+4
        x[240] = abs(hash('i2223_' + str(round(float(row[22]),2)) + '_' + str(round(float(row[23]),2)))) % D
        x[241] = abs(hash('i8384_' + str(round(float(row[83]),2)) + '_' + str(round(float(row[84]),2)))) % D
        x[242] = abs(hash('i113114_' + str(round(float(row[113]),2)) + '_' + str(round(float(row[114]),2)))) % D
        x[243] = abs(hash('i138139_' + str(round(float(row[138]),2)) + '_' + str(round(float(row[139]),2)))) % D

        #+5
        hesh = 1 if round(float(row[22]),2) > round(float(row[23]),2) else (-1 if round(float(row[23]),2) > round(float(row[22]),2) else 0)
        x[244] = abs(hash('i2223_equal_' + str(hesh))) % D
        hesh = 1 if round(float(row[53]),2) > round(float(row[54]),2) else (-1 if round(float(row[54]),2) > round(float(row[53]),2) else 0)
        x[245] = abs(hash('i5354_equal_' + str(hesh))) % D
        hesh = 1 if round(float(row[83]),2) > round(float(row[84]),2) else (-1 if round(float(row[84]),2) > round(float(row[83]),2) else 0)
        x[246] = abs(hash('i8384_equal_' + str(hesh))) % D
        hesh = 1 if round(float(row[113]),2) > round(float(row[114]),2) else (-1 if round(float(row[114]),2) > round(float(row[113]),2) else 0)
        x[247] = abs(hash('i113114_equal_' + str(hesh))) % D
        hesh = 1 if round(float(row[138]),2) > round(float(row[139]),2) else (-1 if round(float(row[139]),2) > round(float(row[138]),2) else 0)
        x[248] = abs(hash('i138139_equal_' + str(hesh))) % D

        #+14
        x[249]= abs(hash('i1_i2_' + row[1] + row[2])) % D
        x[250]= abs(hash('i10_i11_i12_i13_i14_' + row[10] + row[11] + row[12] + row[13] + row[14])) % D
        x[251]= abs(hash('i24_i25_i26_' + row[24] + row[25] + row[26])) % D
        x[252]= abs(hash('i30_i31_i31_i33_' + row[30] + row[31] + row[32] + row[33])) % D
        x[253]= abs(hash('i41_i42_i43_i44_i45_' + row[41] + row[42] + row[43] + row[44] + row[45])) % D
        x[254]= abs(hash('i55_i56_i57_' + row[55] + row[56] + row[57])) % D
        x[255]= abs(hash('i62_i63_' + row[62] + row[63])) % D
        x[256]= abs(hash('i71_i72_i73_i74_i75_' + row[71] + row[72] + row[73] + row[74] + row[75])) % D
        x[257]= abs(hash('i85_i86_i87_' + row[85] + row[86] + row[87])) % D
        x[258]= abs(hash('i92_i93_' + row[92] + row[93])) % D
        x[259]= abs(hash('i101_i102_i103_i104_i105_' + row[101] + row[102] + row[103] + row[104] + row[105])) % D
        x[260]= abs(hash('i115_i116_i117_' + row[115] + row[116] + row[117])) % D
        x[261]= abs(hash('i126_i127_i128_i129_i130_' + row[126] + row[127] + row[128] + row[129] + row[130])) % D
        x[262]= abs(hash('i140_i141_i142_' + row[140] + row[141] + row[142])) % D

        t = 262
        #+60+20+5
        num_cols = [5,6,7,8,9,16,19,21,22,23,28,29,36,37,38,39,40,47,50,52,53,54,59,60,66,67,68,69,70,77,80,82,83,84,89,90,96,97,98,99,100,107,110,112,113,114,119,120,121,122,123,124,125,132,135,137,138,139,144,145]
        int_cols = [15,17,18,46,48,49,53,58,76,78,79,88,106,108,109,118,131,133,134,143]
        zero_cols = [20,51,81,111,136]
        for i in xrange(len(num_cols)):
            t=t+1
            val = float(row[num_cols[i]]) + 10e-15
            val = 0 if val < 0 else round(log(val),2)
            num_hash = 'i' + str(num_cols[i]) + "_numlog_" + str(val)
            x[t] = abs(hash(num_hash)) % D
        for i in xrange(len(int_cols)):
            t=t+1
            val = float(row[int_cols[i]]) + 10e-15
            val = 0 if val < 0 else round(log(val),2)
            num_hash = 'i' + str(int_cols[i]) + "_numlog_" + str(val)
            x[t] = abs(hash(num_hash)) % D
        for i in xrange(len(zero_cols)):
            t=t+1
            val = float(row[zero_cols[i]]) + 10e-15
            val = 0 if val < 0 else round(log(val),2)
            num_hash = 'i' + str(zero_cols[i]) + "_numlog_" + str(val)
            x[t] = abs(hash(num_hash)) % D

        #+60+20+5
        for i in xrange(len(num_cols)):
            t=t+1
            val = round(log(float(row[num_cols[i]]) ** 2 + 10e-15), 2)
            num_hash = 'i' + str(num_cols[i]) + "_numlog_**2_" + str(val)
            x[t] = abs(hash(num_hash)) % D
        for i in xrange(len(int_cols)):
            t=t+1
            val = round(log(float(row[int_cols[i]]) ** 2 + 10e-15), 2)
            num_hash = 'i' + str(int_cols[i]) + "_numlog_**2_" + str(val)
            x[t] = abs(hash(num_hash)) % D
        for i in xrange(len(zero_cols)):
            t=t+1
            val = round(log(float(row[zero_cols[i]]) ** 2 + 10e-15), 2)
            num_hash = 'i' + str(zero_cols[i]) + "_numlog_**2_" + str(val)
            x[t] = abs(hash(num_hash)) % D

        # parse y, if provided
        if label_path:
            # use float() to prevent future type casting, [1:] to ignore id
            y = [float(y) for y in label.readline().split(',')[1:]]
        yield (ID, x, y) if label_path else (ID, x)


# B. Bounded logloss
# INPUT:
#     p: our prediction
#     y: real answer
# OUTPUT
#     bounded logarithmic loss of p given y
def logloss(p, y):
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


# C. Get probability estimation on x
# INPUT:
#     x: features
#     w: weights
# OUTPUT:
#     probability of p(y = 1 | x; w)
def predict(x, w):
    wTx = 0.
    for i in x:  # do wTx
        wTx += w[i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
    return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid


# D. Update given model
# INPUT:
# alpha: learning rate
#     w: weights
#     n: sum of previous absolute gradients for a given feature
#        this is used for adaptive learning rate
#     x: feature, a list of indices
#     p: prediction of our model
#     y: answer
# MODIFIES:
#     w: weights
#     n: sum of past absolute gradients
def update(alpha, w, n, x, p, y):
    for i in x:
        # alpha / sqrt(n) is the adaptive learning rate
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1.
        n[i] += abs(p - y)
        w[i] -= (p - y) * 1. * alpha / sqrt(n[i])


# training and testing #######################################################
start = datetime.now()

# a list for range(0, 33) - 13, no need to learn y14 since it is always 0
K = [k for k in range(33) if k not in [13,32]]

# initialize our model, all 32 of them, again ignoring y14
w = [[0.] * D if k not in [13,32] else None for k in range(33)]
n = [[0.] * D if k not in [13,32] else None for k in range(33)]

loss = 0.
loss_y14 = log(1. - 10**-15)
passNum = 0
lastLoss = 10.
thisLoss = 1.
while (lastLoss - thisLoss) > 0.000001 and passNum < 5:
    lastLoss = thisLoss
    passNum += 1
    for ID, x, y in data(train, label):
        ID = ID + 1700000*(passNum-1)
        # get predictions and train on all labels
        for k in K:
            p = predict(x, w[k])
            update(alpha, w[k], n[k], x, p, y[k])
            loss += logloss(p, y[k])  # for progressive validation
        loss += loss_y14  # the loss of y14, logloss is never zero

        # print out progress, so that we know everything is working
        if ID % 100000 == 0:
            monitor.write('%s\tencountered: %d\tcurrent logloss: %f\n' % (
                datetime.now(), ID, (loss/33.)/ID))
            monitor.flush()

    thisLoss = (loss/32)/ID
    thisFile = './submission#'+str(solution)+'_'+str(passNum)+'.csv'
    with open(thisFile, 'w') as outfile:
        outfile.write('id_label,pred\n')
        for ID, x in data(test):
            predSum = 1.0
            for k in K:
                p = predict(x, w[k])
                outfile.write('%s_y%d,%s\n' % (ID, k+1, str(p)))
                predSum -= p
                if k == 12:
                    outfile.write('%s_y14,0.0\n' % ID)
                if k == 31:
                    p = max(0.01,predSum)
                    outfile.write('%s_y33,%s\n' % (ID, str(p)))

monitor.write('Done, elapsed time: %s\n' % str(datetime.now() - start))
monitor.close()

