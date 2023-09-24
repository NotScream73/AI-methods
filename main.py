import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        df = pd.read_csv('data.csv', sep=';')
        return render_template('index.html', data=df.fillna(''), datatable=df, num_columns=len(df.axes[1]),
                               num_rows=len(df.axes[0]),
                               empty_cells=df.isna().sum(), fill_cells=df.count())
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

if __name__ == '__main__':
    app.run(debug=True)
