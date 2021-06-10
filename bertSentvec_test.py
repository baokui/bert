# export CUDA_VISIBLE_DEVICES=0
from bertSentvec import *
import numpy as np
import random
with open('/search/odin/guobk/data/VpaAllSceneUpgradeData/data_train/test.json','r') as f:
    D = json.load(f)
if 1:
    initial_checkpoint = '/search/odin/guobk/data/bert_semantic/model2/model.ckpt-20000'
    nbSamples = 20000
    Sq = [D[i]['query'] for i in range(nbSamples)]
    Y_qr = sentEmb(Sq, mode='qr',init='train',initial_checkpoint=initial_checkpoint)
    Y_qr = norm(Y_qr)
    Sd = []
    for i in range(nbSamples):
        Sd.extend(D[i]['docs'])
    Sd0 = list(set(Sd))
    Y_dc0 = sentEmb(Sd0, mode='dc', init='train',initial_checkpoint=initial_checkpoint)
    Y_dc0 = norm(Y_dc0)
    D_dc = {Sd0[i]:Y_dc0[i] for i in range(len(Sd0))}
    ndcg_raw = []
    ndcg_new = []
    ndcg_rand = []
    for i in range(nbSamples):
        cg = D[i]['scores']
        Sd = D[i]['docs']
        if cg[0]==1:
            continue
        #Y_dc = sentEmb(Sd, mode='dc', init='train',initial_checkpoint=initial_checkpoint)
        #Y_dc = norm(Y_dc)
        Y_dc = np.array([D_dc[k] for k in Sd])
        y_qr = Y_qr[i]
        p = Y_dc.dot(y_qr)
        cg_new = [(p[i],cg[i]) for i in range(len(cg))]
        cg_new = sorted(cg_new, key=lambda x:-x[0])
        cg_new = [t[1] for t in cg_new]
        # if cg_new[0]==1:
        #     continue
        dcg = sum([cg_new[j]/np.log2(j+2) for j in range(len(cg_new))])
        idcg = D[i]['idcg']
        ndcg = dcg/(0.00001+idcg)
        D[i]['ndcg_new'] = ndcg
        ndcg_new.append(ndcg)
        ndcg_raw.append(D[i]['ndcg_raw'])
        cg_new = [i for i in cg]
        # cg_new1 = cg_new[1:]
        # random.shuffle(cg_new1)
        # cg_new = [cg_new[0]]+cg_new1
        random.shuffle(cg_new)
        dcg = sum([cg_new[j]/np.log2(j+2) for j in range(len(cg_new))])
        idcg = D[i]['idcg']
        ndcg = dcg/(0.00001+idcg)
        ndcg_rand.append(ndcg)
        if i%100==0:
            print(i,np.mean(ndcg_raw),np.mean(ndcg_rand),np.mean(ndcg_new))




path_query = '/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/test_s2v/baseData/query.txt'
path_docs = '/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/test_s2v/baseData/Docs.txt'
with open(path_query,'r') as f:
    S = f.read().strip().split('\n')
Y_qr = sentEmb(S, mode='qr',init='train',initial_checkpoint=initial_checkpoint)
Y_qr = norm(Y_qr)
with open(path_docs,'r') as f:
    Sd = f.read().strip().split('\n')
Y_dc0 = sentEmb(Sd, mode='dc', init='train',initial_checkpoint=initial_checkpoint)
Y_dc0 = norm(Y_dc0)

score = Y_qr.dot(np.transpose(Y_dc0))
idx = np.argsort(-score,axis=0)
R = []
for i in range(len(S)):
    r = [Sd[idx[i][k]] for k in range(10)]
    R.append({'query':S[i],'result':r})

with open('/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/test_s2v/baseData/result_bert_sup.json','w') as f:
    json.dump(R,f,ensure_ascii=False,indent=4)