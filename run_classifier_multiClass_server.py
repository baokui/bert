from run_classifier_multiClass import bert_cls
from flask import Flask, request, Response
from gevent.pywsgi import WSGIServer
from gevent import monkey
import json
import logging
import sys
import requests
monkey.patch_all()
app = Flask(__name__)
L0 = ['使用场景P0', '表达对象P0', '表达者性别倾向P0', '文字风格']
model = bert_cls()
@app.route('/api/bertMultiCLS', methods=['POST'])
def test1():
    r = request.json
    inputStr = r["input"]
    result = {}
    try:
        result,_ = model.predict(inputStr)
        info_msg = 'success'
        err_msg = ''
    except Exception as e:
        #app.logger.error("error:",str(e))
        err_msg = e
        info_msg = 'error'
    response = {'msg':info_msg,'errMsg':err_msg,'result':result}
    app.logger.error('GEN_ERROR\t' + json.dumps(response))
    response_pickled = json.dumps(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")
def demo():
    import pymysql
    url_cls = 'http://10.160.25.112:2023/api/bertMultiCLS'
    conn = pymysql.connect(
        host='mt.tugele.rds.sogou',
        user='tugele_new',
        password='tUgele2017OOT',
        charset='utf8',
        port=3306,
        # autocommit=True,    # 如果插入数据，， 是否自动提交? 和conn.commit()功能一致。
    )
    sqli = "SELECT a.id,a.content,b.secondlySceneId,c.name " \
           "FROM tugele.ns_flx_wisdom_words_new a LEFT JOIN tugele.ns_flx_wisdom_word_scene b ON a.id = b.wordId " \
           "LEFT JOIN tugele.ns_flx_wisdom_scene c ON b.secondlySceneId = c.id " \
           "WHERE a.STATUS = 1 " \
           "AND a.quality = 1 " \
           "AND c.STATUS = 1 " \
           "AND a.isDeleted = 0 " \
           "AND c.isDeleted = 0"
    cur = conn.cursor()
    result = cur.execute(sqli)
    info = cur.fetchall()
    cur.close()
    conn.close()
    S = [info[i][1] for i in range(len(info))]
    L0 = ['使用场景P0', '表达对象P0', '表达者性别倾向P0', '文字风格']
    R = [[t.replace('P0','') for t in L0]]
    for i in range(len(S)):
        r = requests.post(url_cls,json={'input':S[i]}).json()
        R.append([r['result'][t][0] for t in L0]+[S[i]])
        if i%100==0:
            print(i,len(S))
        if i%1000==0:
            with open('tmp_multi.txt','w',encoding='utf-8') as f:
                _ = f.write('\n'.join(['\t'.join(t) for t in R]))


if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

if __name__ == "__main__":
    port = int(sys.argv[1])
    server = WSGIServer(("0.0.0.0", port), app)
    print("Server started")
    server.serve_forever()