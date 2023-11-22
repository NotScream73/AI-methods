import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import re
import seaborn as sns
from matplotlib import pyplot as plt
from io import BytesIO
import base64
import hashlib
from bitarray import bitarray
from sklearn.metrics import r2_score

app = Flask(__name__)

global returnbtn
returnbtn = "<a href='/'>Назад</a"

dataframe = pd.read_csv('data.csv', sep=';')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', data=dataframe.fillna(''), datatable=dataframe, num_columns=len(dataframe.axes[1]),
                               num_rows=len(dataframe.axes[0]),
                               empty_cells=dataframe.isna().sum(), fill_cells=dataframe.count(),
                               task1 = task1, task2 = task2(), task3 = task3(), task4 = task4())
    else:
        data = request.form
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
        return render_template('index.html', data=(dataframe.iloc[from_str:to_str, from_col:to_col]).fillna(''), datatable=dataframe,
                               num_columns=len(dataframe.axes[1]), num_rows=len(dataframe.axes[0]),
                               empty_cells=dataframe.isna().sum(), fill_cells=dataframe.count())

@app.route('/task1')
def task1():
    df = dataframe.copy()
    validvalue(df)
    updf = df.copy()
    tenprocent(updf)
    result = df.groupby(df['Country'])['Valuation ($B) '].agg(['min', 'max', 'mean'])
    result['mean'] = round(result['mean'],2)
    updresult = updf.groupby(updf['Country'])['Valuation ($B) '].agg(['min', 'max', 'mean'])
    updresult['mean'] = round(updresult['mean'],2)
    return render_template('tasks.html', data = result.to_html, updata = updresult.to_html, url="/task?task=1")
@app.route('/task2')
def task2():
    df = dataframe.copy()
    validvalue(df)
    updf = df.copy()
    tenprocent(updf)
    result = df.groupby(df['Founded Year'].dt.year)['Valuation ($B) '].agg(['min', 'max', 'mean'])
    result['mean'] = round(result['mean'],2)
    updresult = updf.groupby(updf['Founded Year'].dt.year)['Valuation ($B) '].agg(['min', 'max', 'mean'])
    updresult['mean'] = round(updresult['mean'],2)
    return render_template('tasks.html', data=result.to_html, updata=updresult.to_html, url="/task?task=2")
@app.route('/task3')
def task3():
    df = dataframe.copy()
    validvalue(df)
    updf = df.copy()
    tenprocent(updf)
    result = df.groupby(df['Number of Employees'])['Valuation ($B) '].agg(['min', 'max', 'mean'])
    result['mean'] = round(result['mean'], 2)
    updresult = updf.groupby(updf['Number of Employees'])['Valuation ($B) '].agg(['min', 'max', 'mean'])
    updresult['mean'] = round(updresult['mean'], 2)
    return render_template('tasks.html', data=result.to_html, updata=updresult.to_html, url="/task?task=3")
@app.route('/task4')
def task4():
    df = dataframe.copy()
    validvalue(df)
    updf = df.copy()
    tenprocent(updf)
    result = df.groupby(df['City'])['Valuation ($B) '].agg(['min', 'max', 'mean'])
    result['mean'] = round(result['mean'], 2)
    updresult = updf.groupby(updf['City'])['Valuation ($B) '].agg(['min', 'max', 'mean'])
    updresult['mean'] = round(updresult['mean'], 2)
    return render_template('tasks.html', data=result.to_html, updata=updresult.to_html, url="/task?task=4")
def validvalue(df):
    df['Valuation ($B) '] = df['Valuation ($B) '].apply(lambda x: re.sub('[^0-9\.]+', '', x))
    df['Valuation ($B) '] = pd.to_numeric(df['Valuation ($B) '])
    df['Total Funding'] = df['Total Funding'].str.replace('$', '').str.replace('M', '').str.replace(',', '')
    df['Total Funding'] = pd.to_numeric(df['Total Funding'])
    df['Founded Year'] = pd.to_datetime(df['Founded Year'], format='%Y')
def tenprocent(updf):
    # Рассчитываем 10% от общего количества строк в DataFrame
    percent_to_fill = 0.1
    num_rows_to_fill = int(len(dataframe) * percent_to_fill)

    # Получаем усредненные значения для числовых столбцов
    num_columns = ['Valuation ($B) ', 'Total Funding']
    num_avg_values = updf[num_columns].mean()

    # Получаем наиболее часто встречающиеся значения для текстовых столбцов
    text_columns = ['Company', 'Country', 'Founded Year', 'State', 'Number of Employees', 'City', 'Industries',
                    'Name of Founders']
    text_most_common_values = updf[text_columns].mode().iloc[0]

    # Дополняем 10% строк DataFrame усредненными значениями и наиболее часто встречающимися значениями

    for i in range(1, num_rows_to_fill):
        # Создаем новую строку для DataFrame
        new_row = pd.Series()
        for column in num_columns:
            new_row[column] = num_avg_values[column]
        for column in text_columns:
            new_row[column] = text_most_common_values[column]
        # Добавляем строку в DataFrame с использованием loc
        updf.loc[len(updf)] = new_row
@app.route('/task')
def task5():
    page = request.args.get('task', default=1, type=int)
    df = dataframe.copy()
    validvalue(df)
    updf = df.copy()
    tenprocent(updf)
    if page == 1:
        result = df[['Country', 'Valuation ($B) ']]
        updf = df.copy()
        tenprocent(updf)
        result2 = updf[['Country', 'Valuation ($B) ']]
        # Строим boxplot с использованием Seaborn
        plt.figure(figsize=(15, 10))
        sns.boxplot(x='Country', y='Valuation ($B) ', data=result)
        plt.title('Valuation Boxplot by Country')
        plt.xlabel('Country')
        plt.ylabel('Valuation')
        # Сохраняем график в байтовом потоке
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        # Кодируем изображение в формат base64
        img_base64 = base64.b64encode(img.getvalue()).decode()
        plt.clf()
        sns.boxplot(x='Country', y='Valuation ($B) ', data=result2)
        img2 = BytesIO()
        plt.savefig(img2, format='png')
        img2.seek(0)
        # Кодируем изображение в формат base64
        upimg_base64 = base64.b64encode(img2.getvalue()).decode()
        # Отправляем шаблон с изображением в HTML
        return render_template('image.html', img_data=img_base64, img_updata=upimg_base64)
    elif page == 2:
        result = df[['Founded Year', 'Valuation ($B) ']]
        updf = df.copy()
        tenprocent(updf)
        result2 = updf[['Founded Year', 'Valuation ($B) ']]
        # Строим boxplot с использованием Seaborn
        plt.figure(figsize=(15, 10))
        sns.boxplot(x='Founded Year', y='Valuation ($B) ', data=result)
        plt.title('Valuation Boxplot by Founded Year')
        plt.xlabel('Founded Year')
        plt.ylabel('Valuation')
        # Сохраняем график в байтовом потоке
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        # Кодируем изображение в формат base64
        img_base64 = base64.b64encode(img.getvalue()).decode()
        plt.clf()
        sns.boxplot(x='Founded Year', y='Valuation ($B) ', data=result2)
        img2 = BytesIO()
        plt.savefig(img2, format='png')
        img2.seek(0)
        # Кодируем изображение в формат base64
        upimg_base64 = base64.b64encode(img2.getvalue()).decode()
        # Отправляем шаблон с изображением в HTML
        return render_template('image.html', img_data=img_base64, img_updata=upimg_base64)
    elif page == 3:
        result = df[['Number of Employees', 'Valuation ($B) ']]
        updf = df.copy()
        tenprocent(updf)
        result2 = updf[['Number of Employees', 'Valuation ($B) ']]
        # Строим boxplot с использованием Seaborn
        plt.figure(figsize=(15, 10))
        sns.boxplot(x='Number of Employees', y='Valuation ($B) ', data=result)
        plt.title('Valuation Boxplot by Number of Employees')
        plt.xlabel('Number of Employees')
        plt.ylabel('Valuation')
        # Сохраняем график в байтовом потоке
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        # Кодируем изображение в формат base64
        img_base64 = base64.b64encode(img.getvalue()).decode()
        plt.clf()
        sns.boxplot(x='Number of Employees', y='Valuation ($B) ', data=result2)
        img2 = BytesIO()
        plt.savefig(img2, format='png')
        img2.seek(0)
        # Кодируем изображение в формат base64
        upimg_base64 = base64.b64encode(img2.getvalue()).decode()
        # Отправляем шаблон с изображением в HTML
        return render_template('image.html', img_data=img_base64, img_updata=upimg_base64)
    elif page == 4:
        result = df[['City', 'Valuation ($B) ']]
        updf = df.copy()
        tenprocent(updf)
        result2 = updf[['City', 'Valuation ($B) ']]
        # Строим boxplot с использованием Seaborn
        plt.figure(figsize=(40, 10))
        sns.boxplot(x='City', y='Valuation ($B) ', data=result)
        plt.title('Valuation Boxplot by City')
        plt.xlabel('City')
        plt.ylabel('Valuation')
        # Сохраняем график в байтовом потоке
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        # Кодируем изображение в формат base64
        img_base64 = base64.b64encode(img.getvalue()).decode()
        plt.clf()
        sns.boxplot(x='City', y='Valuation ($B) ', data=result2)
        img2 = BytesIO()
        plt.savefig(img2, format='png')
        img2.seek(0)
        # Кодируем изображение в формат base64
        upimg_base64 = base64.b64encode(img2.getvalue()).decode()
        # Отправляем шаблон с изображением в HTML
        return render_template('image.html', img_data=img_base64, img_updata=upimg_base64)



class BloomFilter:
    def __init__(self, size, hash_functions):
        self.size = size
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)
        self.hash_functions = hash_functions

    def _hash(self, item, function):
        return int(hashlib.sha256(item.encode()).hexdigest(), 16) % self.size

    def add(self, item):
        for hash_func in self.hash_functions:
            index = self._hash(item, hash_func) % self.size
            self.bit_array[index] = 1

    def contains(self, item):
        for hash_func in self.hash_functions:
            index = self._hash(item, hash_func) % self.size
            if not self.bit_array[index]:
                return False
        return True


keywords_links = [
    ({"Company", "Valuation", "Country", "State", "City", "Industries", "Founded Year", "Name of Founders", "Total Funding", "Number of Employees"}, "https://www.kaggle.com/datasets/ankanhore545/100-highest-valued-unicorns"),
    ({"Insider Trading", "Relationship", "Date", "Transaction", "Cost", "Shares", "Value", "Shares Total", "SEC Form 4"}, "https://www.kaggle.com/datasets/ilyaryabov/tesla-insider-trading"),
    ({"NASA", "est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance", "orbiting_body", "sentry_object", "absolute_magnitude", "hazardous"}, "https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects"),
    ({"авто", "автомобиль", "car", "price", "стоимость", "производитель", "manufacturer"}, "https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge"),
    ({"Country", "Population", "Continent", "Capital", "Yearly Change", "Land Area", "Fertility","Density"}, "https://www.kaggle.com/datasets/muhammedtausif/world-population-by-countries"),
    ({"Health", "Diabetes", "India"}, "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"),
    ({"age","sex","bmi"}, "https://www.kaggle.com/datasets/mirichoi0218/insurance"),
    ({"store", "store area", "items", "available", "daily customer", "sales"}, "https://www.kaggle.com/datasets/surajjha101/stores-area-and-sales-data"),
    ({"Name" , "Networth", "Source"}, "https://www.kaggle.com/datasets/surajjha101/forbes-billionaires-data-preprocessed"),
    ({"heart_disease" , "bmi", "stroke"}, "https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset"),
]

hash_functions = ["sha256"]

filter_size = 1000
bloom_filter = BloomFilter(filter_size, hash_functions)

for keywords, link in keywords_links:
    for keyword in keywords:
        bloom_filter.add(keyword.lower())

@app.route('/search')
def searc():
    return render_template('search.html')

@app.route('/search', methods=['POST'])
def search():
    user_input = request.form['search_query'].lower().split(', ')
    result_link = []
    for keyword in user_input:
        if bloom_filter.contains(keyword):
            for keywords, link in keywords_links:
                if keyword in list(map(str.lower, keywords)):
                    result_link.append(link)
    return render_template('search.html', result=set(result_link))


def create_linear_model(data):
    # ИМТ
    X = data.iloc[:, 10].values
    # АД
    Y = data.iloc[:, 8].values
    n = data.shape[0]

    # Разделение данных на обучающий и тестовый наборы
    n_test = int(n * 0.05)
    n_train = n - n_test
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_test, Y_test = X[n_train:], Y[n_train:]

    sumY_train = sum(Y_train)
    sumX_train = sum(X_train)

    sumXY_train = sum(X_train * Y_train)
    sumXX_train = sum(X_train * X_train)

    b1 = (sumXY_train - (sumY_train * sumX_train) / n_train) / (sumXX_train - sumX_train * sumX_train / n_train)
    b0 = (sumY_train - b1 * sumX_train) / n_train

    # Построение модели на обучающем наборе
    # plt.scatter(X_train, Y_train, alpha=0.8)
    # plt.axline(xy1=(0, b0), slope=b1, color='r', label=f'$y = {b1:.5f}x {b0:+.5f}$')

    # Оценка производительности модели на тестовом наборе
    Y_pred = b0 + b1 * X_test
    first_half = sum((Y_pred - Y_test.mean()) ** 2)
    second_half = sum((Y_test - Y_pred) ** 2) + first_half

    r2 = r_squared(Y_test, Y_pred)
    print(f"Кэф по странной формуле из вики: {first_half/second_half}")
    print(f"Истинный кэф по вики: {r2}")
    print(f"Кэф из библы: {r2_score(Y_test, Y_pred)}")

    # plt.scatter(X_test, Y_test, alpha=0.8, color='g')
    # plt.legend()
    # plt.show()
    return r2
def r_squared(y_true, y_pred):
    # Вычисляем среднее значение целевой переменной
    mean_y_true = np.mean(y_true)

    # Вычисляем сумму квадратов отклонений от среднего
    ss_total = np.sum((y_true - mean_y_true) ** 2)

    # Вычисляем сумму квадратов остатков
    ss_residual = np.sum((y_true - y_pred) ** 2)

    # Вычисляем коэффициент детерминации
    return 1 - (ss_residual / ss_total)

# Чтение данных из файла
data = pd.read_csv('data.csv', delimiter=';')

# Убираем все лишние символы из первого столбца и оставляем только числа
data['Total Funding'] = data['Total Funding'].replace('[\$,M]', '', regex=True).astype(float)

# Функция для обработки столбца 'Number of Employees'
def process_employees(value):
    if '-' in value:
        # Разделяем значение на два числа и убираем запятые
        start, end = map(int, value.replace(',', '').split('-'))
        # Вычисляем среднее значение из двух чисел
        return (start + end) // 2
    elif 'No Data' in value:
        return None
    else:
        # Убираем десятичную точку и преобразуем в целое число
        return int(value.replace('.', ''))

# Применяем функцию к столбцу 'Number of Employees' и создаем новый столбец 'Employees'
data['Employees'] = data['Number of Employees'].apply(process_employees)

# Удаляем строки, где 'Number of Employees' был 'No Data'
data = data.dropna(subset=['Employees'])
create_linear_model(data)
if __name__ == '__main__':
    app.run(debug=True)