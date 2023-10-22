from django.shortcuts import render
import joblib
import numpy as np
# Create your views here.

model = joblib.load('model.joblib')

def index(request):
    print('index')
    return render(request, 'index.html')

def predict(request):
    sepal_length = request.GET.get('sepal_length')
    sepal_width = request.GET.get('sepal_width')
    petal_length = request.GET.get('petal_length')
    petal_width = request.GET.get('petal_width')

    data = np.array([[sepal_length, sepal_width,
                      petal_length, petal_width]])

    pred = model.predict(data)

    species_prediction = ''
    if pred == 0:
        species_prediction = 'setosa'
    elif pred == 1:
        species_prediction = 'versicolor'
    elif pred == 2:
        species_prediction = 'virginica'

    return render(request, 'predict.html', {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width,
        'species_prediction': species_prediction
    })

