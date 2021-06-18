from run_classifier_application import bert_cls
import numpy as np
import os
import sys
path_source,path_target,gpu = sys.argv[1:]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
Batch = 10000
with open(path_source,'r') as f:
    C = f.read().strip().split('\n')
f_w = open(path_target,'a+')
path_model = "/search/odin/guobk/data/AiWriter/model/text_quality_multi/"
num_labels = 4
max_seq_length = 128
model = bert_cls(path_model, num_labels, max_seq_length)
id0 = 0
n = 0
while id0*Batch<len(C):
    S = [c.split('\t')[1] for c in C[id0*Batch:(id0+1)*Batch]]
    p = model.predict_batch(S, log=False)
    y = np.argmax(p,axis=1)
    R = ['\t'.join([S[i],'%0.4f'%p[i][0]]) for i in range(len(S)) if y[i]==0]
    f_w.write('\n'.join(R)+'\n')
    n+=len(R)
    print(id0,int(len(C)/Batch),n)
    id0 += 1
f_w.close()

