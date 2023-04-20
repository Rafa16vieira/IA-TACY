import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Carregar o conjunto de dados a partir de um arquivo CSV, transformar dados textuais, 
csv = pd.read_csv("predios.csv", sep=",")
le = LabelEncoder()
csv["location"] = le.fit_transform(csv["location"])
dados = csv.values
atributos = dados[:,0:4]
classificadores = dados[:,4]



# Criar um modelo de regressão linear simples
modelo = LinearRegression()

# Treinar o modelo com os dados de entrada e os resultados
modelo.fit(atributos, classificadores)

# modelo para fazer previsões em novos dados de entrada (quartos, m², região (0=leste, 1=norte, 2=sul, 3=oeste), idade)
novos_dados = [[1, 80, 0, 15], [4, 200, 1, 1]]
previsoes = modelo.predict(novos_dados)

# Exibir as previsões
print(previsoes)


#print(atributos)
#print(classificadores)