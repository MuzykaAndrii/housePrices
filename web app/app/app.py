from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    else:
        with open("house_prices_pipeline.pickle", "rb") as infile:
            pipeline = pickle.load(infile)

        df = pd.DataFrame(
            {
                'MSSubClass': [request.form['MSSubClass']],
                'MSZoning': [request.form['MSZoning']],
                'LotArea': [request.form['LotArea']],
                'Alley': [request.form['Alley']],
                'LotShape': [request.form['LotShape']],
                'Condition1': [request.form['Condition1']],
                'BldgType': [request.form['BldgType']],
                'OverallCond': [request.form['OverallCond']],
                'YearBuilt': [request.form['YearBuilt']],
                'RoofStyle': [request.form['RoofStyle']],
                'RoofMatl': [request.form['RoofMatl']],
                'ExterCond': [request.form['ExterCond']],
                'Heating': [request.form['Heating']],
                'Electrical': [request.form['Electrical']]
            }
        )

        prediction = pipeline.predict(df)
        
        return render_template('result.html', result=prediction)