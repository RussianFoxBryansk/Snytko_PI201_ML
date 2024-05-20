from requests import get
sepal_length = input('Введите sepal_length = ')
sepal_width = input('Введите sepal_width = ')
petal_length = input('Введите petal_length = ')
print(get(f'http://localhost:5000/api?list1={sepal_length}&list2={sepal_length}&list3={sepal_length}').json())