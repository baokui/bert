#!/usr/bin/python
# -*- coding: utf-8 -*-
import jieba_fast as jieba
import os
import sys
def textSeg(sourefile0,targetfile0):
    targetfile = targetfile0 + '/sents.txt'
    f_w = open(targetfile, 'w+')
    k = 0
    for p in range(5):
        sourefile = sourefile0+'/part-0000'+str(p)
        f_r = open(sourefile,'r')
        for line in f_r:
            line = line.strip().lower()
            s = line.split('\t')
            if len(s)!=3:
                continue
            words = " ".join(jieba.lcut(s[2], HMM=True))
            f_w.write(words+'\n')
            k+=1
            if k%10000==0:
                print('write %d lines'%k)
        f_r.close()
        #os.remove(sourefile)
    f_w.close()
def main(sourefile):
    if not os.path.exists(sourefile+'-seg'):
        os.mkdir(sourefile+'-seg')
    textSeg(sourefile,sourefile+'-seg')
    #os.remove(sourefile)
if __name__=='__main__':
    sourcefile = sys.argv[1]
    main(sourcefile)