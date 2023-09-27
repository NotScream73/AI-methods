import pandas as pd
from flask import Flask, render_template, request
import re

app = Flask(__name__)
global returnbtn
returnbtn = "<a href='/'>Назад</a"
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        df = pd.read_csv('data.csv', sep=';')
        return render_template('index.html', data=df.fillna(''), datatable=df, num_columns=len(df.axes[1]),
                               num_rows=len(df.axes[0]),
                               empty_cells=df.isna().sum(), fill_cells=df.count(),
                               task1 = task1, task2 = task2(), task3 = task3(), task4 = task4())
    else:
        data = request.form
        df = pd.read_csv('data.csv', sep=';')
        from_str = 0
        to_str = 100
        from_col = 0
        to_col = 100
        if len(data['minrow']) != 0:
            from_str = int(data['minrow'])
        if len(data['maxrow']) != 0:
            to_str = int(data['maxrow'])
        if len(data['mincol']) != 0:
            from_col = int(data['mincol'])
        if len(data['maxcol']) != 0:
            to_col = int(data['maxcol'])
        return render_template('index.html', data=(df.iloc[from_str:to_str, from_col:to_col]).fillna(''), datatable=df,
                               num_columns=len(df.axes[1]), num_rows=len(df.axes[0]),
                               empty_cells=df.isna().sum(), fill_cells=df.count())

@app.route('/task1')
def task1():
    df = pd.read_csv('data.csv', sep=';')
    df['Valuation ($B) '] = df['Valuation ($B) '].apply(lambda x: re.sub('[^0-9\.]+', '', x))
    df['Valuation ($B) '] = pd.to_numeric(df['Valuation ($B) '])
    result = df.groupby('Country').agg({'Valuation ($B) ': ['min', 'max', 'mean']})
    return result.to_html() + returnbtn
@app.route('/task2')
def task2():
    df = pd.read_csv('data.csv', sep=';')
    df['Founded Year'] = pd.to_datetime(df['Founded Year'], format='%Y')
    df['Valuation ($B) '] = df['Valuation ($B) '].apply(lambda x: re.sub('[^0-9\.]+', '', x))
    df['Valuation ($B) '] = pd.to_numeric(df['Valuation ($B) '])
    result = df.groupby(df['Founded Year'].dt.year)['Valuation ($B) '].agg(['min', 'max', 'mean'])
    result['mean'] = round(result['mean'],2)
    return result.to_html() + returnbtn
@app.route('/task3')
def task3():
    df = pd.read_csv('data.csv', sep=';')
    df['Valuation ($B) '] = df['Valuation ($B) '].apply(lambda x: re.sub('[^0-9\.]+', '', x))
    df['Valuation ($B) '] = pd.to_numeric(df['Valuation ($B) '])
    result = df.groupby('Number of Employees').agg({'Valuation ($B) ': ['min', 'max', 'mean']})
    return result.to_html() + returnbtn
@app.route('/task4')
def task4():
    df = pd.read_csv('data.csv', sep=';')
    df['Valuation ($B) '] = df['Valuation ($B) '].apply(lambda x: re.sub('[^0-9\.]+', '', x))
    df['Valuation ($B) '] = pd.to_numeric(df['Valuation ($B) '])
    result = df.groupby('City').agg({'Valuation ($B) ': ['min', 'max', 'mean']})
    return result.to_html() + returnbtn

if __name__ == '__main__':
    app.run(debug=True)