import numpy as np
import sys
sys.path.append('/Users/kristen.severson.ibm/PycharmProjects/cLVM/contrastive-LVM/')
from clvm_tfp import clvm
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from contrastive import CPCA

#Load Data
data = joblib.load(filename='RNASeq_Processed.pkl')
A = np.array(data['A'])
B = np.array(data['B'])
labels = np.array(data['labels'])