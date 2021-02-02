from run_classifier_multiClass1 import bert_cls
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
    app.logger.error('GEN_ERROR\t' + json.dumps(response,ensure_ascii=False,indent=4))
    response_pickled = json.dumps(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")
def demo():
    with open('/search/odin/guobk/vpa/vpa-studio-research/labelClassify/DataLabel/test.txt','r') as f:
        S = f.read().strip().split('\n')
    D = {k:[] for k in L0}
    for i in range(len(S)):
        inputStr = S[i].split('\t')[0]
        _, r = model.predict(inputStr)
        for j in range(len(L0)):
            D[L0[j]].append(['%0.6f'%t for t in list(r[j])])
    for k in L0:
        with open('model/label1/all/result_test_{}.txt'.format(k),'w') as f:
            f.write('\n'.join(['\t'.join(t) for t in D[k]]))

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

if __name__ == "__main__":
    port = int(sys.argv[1])
    server = WSGIServer(("0.0.0.0", port), app)
    print("Server started")
    server.serve_forever()