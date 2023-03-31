# import class AIGDrug from AIGdrugbytarget.py as AIG
import AIGdrugbytarget as AIG
import pandas as pd
import datetime

from flask import Flask, render_template, request, redirect, url_for, session, flash

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/input')
def predict():
    aig =  AIG.AIGDrug('B-lymphocyte antigen CD20')
    target = list(['Insulin-like growth factor 1 receptor',
       'Interferon alpha/beta receptor 1', 'B-lymphocyte antigen CD20',
       'Low affinity immunoglobulin gamma Fc region receptor II-c',
       'Interleukin-23 subunit alpha'])
    df_history = pd.read_csv('/home/herutriana44/Documents/riset/bioinformatika/BioDataset/DrugBank Dataset/history_generated.csv')
    # dataframe to dictionary
    df_history = df_history.to_dict('records')
    df_history = pd.read_csv('static/dataset/history_generated.csv')
    print(df_history.columns)
    df_history['time'] = pd.to_datetime(df_history['time'])
    df_history = df_history.sort_values(by=['time'], ascending=False)
    df_history = df_history.to_dict('records')
    return render_template('input.html', target=target, history=df_history, lenght=len(df_history))

@app.route('/result', methods=['POST','GET'])
def generate():
    if(request.method == 'POST'):
        Target = request.form.getlist('Target')
        # dapatkan value dari dropdown Target
        print("Target: ", end="")
        print(Target)
        aig =  AIG.AIGDrug(Target[0])
        target = list(['Insulin-like growth factor 1 receptor',
       'Interferon alpha/beta receptor 1', 'B-lymphocyte antigen CD20',
       'Low affinity immunoglobulin gamma Fc region receptor II-c',
       'Interleukin-23 subunit alpha'])
        # urutkan target secara ascending
        target = target.sort()
        gen, score, MW, LogP, HBD, HBA, TPSA = aig.GANofVarian()
        result = {
            'gen': gen,
            'score': score,
            'MW': MW,
            'LogP': LogP,
            'HBD': HBD,
            'HBA': HBA,
            'TPSA': TPSA
        }
        inp = [gen, Target[0], score, MW, LogP, HBA, HBD, TPSA, datetime.datetime.now()]
        df_history = pd.read_csv('static/dataset/history_generated.csv')
        df_history = df_history.append(pd.DataFrame([inp], columns=['Sequence', 'target', 'BioAS', 'MW', 'LogP', 'HBA', 'HBD', 'TPSA','time']), ignore_index=True)
        df_history.to_csv('static/dataset/history_generated.csv', index=False)
        
        return render_template('result.html', result=[result], target=target, lenght=len([result]))

if __name__ == '__main__':
    app.run(debug=False, port=4000)