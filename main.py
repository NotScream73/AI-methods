from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    datawithnan = pd.read_csv('data.csv', delimiter=";")
    # Подсчет пустых ячеек в каждом столбце
    empty_cells = datawithnan.isnull().sum()
    fill_cells = datawithnan.notnull().sum()

    # Подсчет количества столбцов
    num_columns = int(datawithnan.shape[1])

    # Подсчет количества строк
    num_rows = int(datawithnan.shape[0])
    data = pd.read_csv('data.csv',delimiter=";", keep_default_na=False)
    minrow = request.form['minrow'] if request.form.__len__() > 0 else 0
    maxrow = request.form['maxrow'] if request.form.__len__() > 0 else num_rows
    mincol = request.form['mincol'] if request.form.__len__() > 0 else 0
    maxcol = request.form['maxcol'] if request.form.__len__() > 0 else num_columns
    if request.method == 'POST':
        data = data.iloc[int(minrow):int(maxrow), int(mincol):int(maxcol)]
        empty_cells = datawithnan.iloc[int(minrow):int(maxrow), int(mincol):int(maxcol)].isnull().sum()
        fill_cells = datawithnan.iloc[int(minrow):int(maxrow), int(mincol):int(maxcol)].notnull().sum()
        print(empty_cells)
    else:
        data = pd.read_csv('data.csv', delimiter=";", keep_default_na=False)

    return render_template('index.html', data=data,datawithnan=datawithnan, num_columns = num_columns, num_rows= num_rows, empty_cells = empty_cells, fill_cells=fill_cells)

if __name__ == '__main__':
    app.run(debug=True)