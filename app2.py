from requests import get
sepal_length = input('Введите sepal_length = ')
sepal_width = input('Введите sepal_width = ')
petal_length = input('Введите petal_length = ')
print(get('http://localhost:5000/api_v2', json={'list1':sepal_length,'list2':sepal_width, 'list3':petal_length}).json())
