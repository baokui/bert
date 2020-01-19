#!/usr/bin/python
# -*- coding: utf-8 -*-
import jieba_fast as jieba
import os
import sys
def textSeg(sourefile,targetfile):
    f_r = open(sourefile,'r')
    f_w = open(targetfile,'w+')
    for line in f_r:
        line = line.strip()
        s = line.split('\t')
        if len(s)!=3:
            continue
        words = " ".join(jieba.lcut(s[2], HMM=True))
        f_w.write(words.encode('utf-8')+'\n')
    f_w.close()
    f_r.close()
def main(sourefile):
    textSeg(sourefile,sourefile+'-seg')
    os.remove(sourefile)
if __name__=='__main__':
    sourcefile = sys.argv[1]
    main(sourcefile)