## PREVISÃO DO SCORE DE CRÉDITO COM REGRESSÃO LINEAR
### Vamos criar um modelo predito de machine learning para prever o score de crédito do cliente:
    - Analise Exploratória e Gráficos<br>
    - Tratamento de dados missing <br>
    - Tratamento de outliers <br>
    - OneHotEncoding <br>
    - Engenharia de Atributos <br>
    - Tratamento de dados <br>
    - Normalização de dados <br>
    - Criação, teste e validação de um modelo de machine learning


```python
# A primeira coisa que temos que fazer é importar os pacotes que iremos utilizar.
# Obs.: Pacotes do Python são conjuntos de funcionalidades disponíveis da ferramenta.

#Pandas: Possui inúmeras funções e comandos para importar arquivos, analisar dados, tratar dados, etc.
import pandas as pd

#Matplotlib: Possui uma série de funções e comandos para exibição de gráficos
import matplotlib.pyplot as plt

#Seaborn: Possui uma série de funções e comandos para exibição de gráficos (Visualizações mais robustas do que o Matplotlib)
import seaborn as sns

#Numpy: Possui uma série de funções e comandos para trabalharmos com números de forma em geral(formatação, calculos, etc)
import numpy as np

#Warnings: Possui detalhes sobre os avisos e alertas que aparecem, porém podemos utiliza-lo também para que os alertas de
#futuras atualizações e metodos depreciados não sejam exibidos
import warnings
warnings.filterwarnings("ignore") 


from sklearn.model_selection import train_test_split # Utilizado para separar dados de treino e teste
from sklearn.preprocessing import StandardScaler # Utilizado para fazer a normalização dos dados
from sklearn.preprocessing import MinMaxScaler # Utilizado para fazer a normalização dos dados
from sklearn.preprocessing import LabelEncoder # Utilizado para fazer o OneHotEncoding
from sklearn.linear_model import LinearRegression # Algoritmo de Regressão Linear
from sklearn.metrics import r2_score # Utilizado para medir a acuracia do modelo preditivo


#Comando para exibir todas colunas do arquivo
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```

## Importação dos dados, Analise Exploratória e Tratamento de Dados


```python
#Comando utilizado para carregar o arquivo e armazena-lo como um DataFrame do Pandas
#Um DataFrame do Pandas é como se fosse uma planilha do Excel, onde podemos tratar linhas e colunas.
df_dados = pd.read_excel("dados_credito.xlsx")
```


```python
#Comando utilizado para verificar a quantidade de linhas e colunas do arquivo
#Colunas também são chamadas de variáveis.
df_dados.shape
```




    (10476, 17)




```python
#Comando utilizado para verificar as linhas iniciais do DataFrame
df_dados.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CODIGO_CLIENTE</th>
      <th>UF</th>
      <th>IDADE</th>
      <th>ESCOLARIDADE</th>
      <th>ESTADO_CIVIL</th>
      <th>QT_FILHOS</th>
      <th>CASA_PROPRIA</th>
      <th>QT_IMOVEIS</th>
      <th>VL_IMOVEIS</th>
      <th>OUTRA_RENDA</th>
      <th>OUTRA_RENDA_VALOR</th>
      <th>TEMPO_ULTIMO_EMPREGO_MESES</th>
      <th>TRABALHANDO_ATUALMENTE</th>
      <th>ULTIMO_SALARIO</th>
      <th>QT_CARROS</th>
      <th>VALOR_TABELA_CARROS</th>
      <th>SCORE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>SP</td>
      <td>19</td>
      <td>Superior Cursando</td>
      <td>Solteiro</td>
      <td>0</td>
      <td>Não</td>
      <td>0</td>
      <td>0</td>
      <td>Não</td>
      <td>0</td>
      <td>8</td>
      <td>Sim</td>
      <td>1800</td>
      <td>0</td>
      <td>0</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>MG</td>
      <td>23</td>
      <td>Superior Completo</td>
      <td>Solteiro</td>
      <td>1</td>
      <td>Não</td>
      <td>0</td>
      <td>0</td>
      <td>Não</td>
      <td>0</td>
      <td>9</td>
      <td>Não</td>
      <td>4800</td>
      <td>1</td>
      <td>50000</td>
      <td>18.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>SC</td>
      <td>25</td>
      <td>Segundo Grau Completo</td>
      <td>Casado</td>
      <td>0</td>
      <td>Sim</td>
      <td>1</td>
      <td>220000</td>
      <td>Não</td>
      <td>0</td>
      <td>18</td>
      <td>Sim</td>
      <td>2200</td>
      <td>2</td>
      <td>30000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>PR</td>
      <td>27</td>
      <td>Superior Cursando</td>
      <td>Casado</td>
      <td>1</td>
      <td>Sim</td>
      <td>0</td>
      <td>0</td>
      <td>Não</td>
      <td>0</td>
      <td>22</td>
      <td>Não</td>
      <td>3900</td>
      <td>0</td>
      <td>0</td>
      <td>28.666667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>RJ</td>
      <td>28</td>
      <td>Superior Completo</td>
      <td>Divorciado</td>
      <td>2</td>
      <td>Não</td>
      <td>1</td>
      <td>370000</td>
      <td>Não</td>
      <td>0</td>
      <td>30</td>
      <td>Sim</td>
      <td>NaN</td>
      <td>1</td>
      <td>35000</td>
      <td>34.166667</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Comando utilizado para verificar as linhas finais do DataFrame
df_dados.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CODIGO_CLIENTE</th>
      <th>UF</th>
      <th>IDADE</th>
      <th>ESCOLARIDADE</th>
      <th>ESTADO_CIVIL</th>
      <th>QT_FILHOS</th>
      <th>CASA_PROPRIA</th>
      <th>QT_IMOVEIS</th>
      <th>VL_IMOVEIS</th>
      <th>OUTRA_RENDA</th>
      <th>OUTRA_RENDA_VALOR</th>
      <th>TEMPO_ULTIMO_EMPREGO_MESES</th>
      <th>TRABALHANDO_ATUALMENTE</th>
      <th>ULTIMO_SALARIO</th>
      <th>QT_CARROS</th>
      <th>VALOR_TABELA_CARROS</th>
      <th>SCORE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10471</th>
      <td>10472</td>
      <td>PR</td>
      <td>51</td>
      <td>Superior Completo</td>
      <td>Solteiro</td>
      <td>1</td>
      <td>Não</td>
      <td>0</td>
      <td>0</td>
      <td>Não</td>
      <td>0</td>
      <td>9</td>
      <td>Não</td>
      <td>4800</td>
      <td>1</td>
      <td>50000</td>
      <td>18.000000</td>
    </tr>
    <tr>
      <th>10472</th>
      <td>10473</td>
      <td>SP</td>
      <td>48</td>
      <td>Segundo Grau Completo</td>
      <td>Casado</td>
      <td>0</td>
      <td>Sim</td>
      <td>1</td>
      <td>220000</td>
      <td>Não</td>
      <td>0</td>
      <td>18</td>
      <td>Sim</td>
      <td>2200</td>
      <td>2</td>
      <td>30000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>10473</th>
      <td>10474</td>
      <td>RJ</td>
      <td>51</td>
      <td>Superior Cursando</td>
      <td>Casado</td>
      <td>1</td>
      <td>Sim</td>
      <td>0</td>
      <td>0</td>
      <td>Não</td>
      <td>0</td>
      <td>22</td>
      <td>Não</td>
      <td>3900</td>
      <td>0</td>
      <td>0</td>
      <td>28.666667</td>
    </tr>
    <tr>
      <th>10474</th>
      <td>10475</td>
      <td>RJ</td>
      <td>48</td>
      <td>Superior Completo</td>
      <td>Divorciado</td>
      <td>2</td>
      <td>Não</td>
      <td>1</td>
      <td>370000</td>
      <td>Não</td>
      <td>0</td>
      <td>30</td>
      <td>Sim</td>
      <td>NaN</td>
      <td>1</td>
      <td>35000</td>
      <td>34.166667</td>
    </tr>
    <tr>
      <th>10475</th>
      <td>10476</td>
      <td>PR</td>
      <td>51</td>
      <td>Segundo Grau Completo</td>
      <td>Divorciado</td>
      <td>0</td>
      <td>Não</td>
      <td>0</td>
      <td>0</td>
      <td>Não</td>
      <td>0</td>
      <td>14</td>
      <td>Sim</td>
      <td>3100</td>
      <td>2</td>
      <td>40000</td>
      <td>39.666667</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Comando utilizado para verificar informações sobre os dados(Tipo de variáveis, Variáveis, Quantidade de registros, etc)

# A variavel CODIGO_CLIENTE poderá ser excluída
# As variaveis UF, ESCOLARIDADE, CASA_PROPRIA, OUTRA_RENDA, TRABALHANDO_ATUALMENTE e ESTADO_CIVIL --> OneHotEncoding
# A variavel ULTIMO_SALARIO está como STRING e precisa ser NUMERICA

df_dados.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10476 entries, 0 to 10475
    Data columns (total 17 columns):
     #   Column                      Non-Null Count  Dtype  
    ---  ------                      --------------  -----  
     0   CODIGO_CLIENTE              10476 non-null  int64  
     1   UF                          10476 non-null  object 
     2   IDADE                       10476 non-null  int64  
     3   ESCOLARIDADE                10476 non-null  object 
     4   ESTADO_CIVIL                10476 non-null  object 
     5   QT_FILHOS                   10476 non-null  int64  
     6   CASA_PROPRIA                10476 non-null  object 
     7   QT_IMOVEIS                  10476 non-null  int64  
     8   VL_IMOVEIS                  10476 non-null  int64  
     9   OUTRA_RENDA                 10476 non-null  object 
     10  OUTRA_RENDA_VALOR           10476 non-null  int64  
     11  TEMPO_ULTIMO_EMPREGO_MESES  10476 non-null  int64  
     12  TRABALHANDO_ATUALMENTE      10476 non-null  object 
     13  ULTIMO_SALARIO              10474 non-null  object 
     14  QT_CARROS                   10476 non-null  int64  
     15  VALOR_TABELA_CARROS         10476 non-null  int64  
     16  SCORE                       10476 non-null  float64
    dtypes: float64(1), int64(9), object(7)
    memory usage: 1.4+ MB



```python
# Vamos excluir a variavel CODIGO_CLIENTE
df_dados.drop('CODIGO_CLIENTE', axis=1, inplace=True)
```


```python
# Dessa forma podemos agrupar os valores e identificar se há algum valor discrepante.
# Observe que há um valor que foi inserido como "SEM DADOS"
df_dados.groupby(['ULTIMO_SALARIO']).size()
```




    ULTIMO_SALARIO
    1800         846
    2200         792
    3100         792
    3900         792
    4500         468
    4800         792
    5300         522
    6100         522
    6800         611
    9000         522
    9800         468
    11500        790
    13000        522
    15000        522
    17500        522
    18300        522
    22000        468
    SEM DADOS      1
    dtype: int64




```python
# Aqui poderíamos resolver de duas formas.

# A primeira forma seria excluir todo o registro, mas estariamos perdendo dados.
#df_dados.drop(df_dados.loc[df_dados['VALOR']=='SEM VALOR'].index, inplace=True)


# A segunda forma seria verificar o valor médio ou da mediana deste modelo e substituir a palavra SEM VALOR para um valor médio.
df_dados.loc[df_dados['ULTIMO_SALARIO'] == 'SEM DADOS']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UF</th>
      <th>IDADE</th>
      <th>ESCOLARIDADE</th>
      <th>ESTADO_CIVIL</th>
      <th>QT_FILHOS</th>
      <th>CASA_PROPRIA</th>
      <th>QT_IMOVEIS</th>
      <th>VL_IMOVEIS</th>
      <th>OUTRA_RENDA</th>
      <th>OUTRA_RENDA_VALOR</th>
      <th>TEMPO_ULTIMO_EMPREGO_MESES</th>
      <th>TRABALHANDO_ATUALMENTE</th>
      <th>ULTIMO_SALARIO</th>
      <th>QT_CARROS</th>
      <th>VALOR_TABELA_CARROS</th>
      <th>SCORE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10459</th>
      <td>RJ</td>
      <td>45</td>
      <td>Superior Cursando</td>
      <td>Solteiro</td>
      <td>1</td>
      <td>Sim</td>
      <td>1</td>
      <td>185000</td>
      <td>Sim</td>
      <td>3000</td>
      <td>19</td>
      <td>Sim</td>
      <td>SEM DADOS</td>
      <td>0</td>
      <td>0</td>
      <td>45.166667</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Agora substituimos a palavra SEM VALOR por um valor NULO
df_dados.replace('SEM DADOS',np.nan, inplace = True)
```


```python
# Em seguida convertemos o campo em float
df_dados['ULTIMO_SALARIO'] = df_dados['ULTIMO_SALARIO'].astype(np.float64)
```


```python
# Comando utilizado para avaliar se alguma variável possui valor nulo ou chamado de valores missing ou NAN (Not Available)
# A variavel ULTIMO_SALARIO possui valores NULOS e precisaremos trata-los
df_dados.isnull().sum()
```




    UF                            0
    IDADE                         0
    ESCOLARIDADE                  0
    ESTADO_CIVIL                  0
    QT_FILHOS                     0
    CASA_PROPRIA                  0
    QT_IMOVEIS                    0
    VL_IMOVEIS                    0
    OUTRA_RENDA                   0
    OUTRA_RENDA_VALOR             0
    TEMPO_ULTIMO_EMPREGO_MESES    0
    TRABALHANDO_ATUALMENTE        0
    ULTIMO_SALARIO                3
    QT_CARROS                     0
    VALOR_TABELA_CARROS           0
    SCORE                         0
    dtype: int64




```python
# Aqui atualizamos o valor conforme a mediana daquele modelo
df_dados['ULTIMO_SALARIO'] = df_dados['ULTIMO_SALARIO'].fillna((df_dados['ULTIMO_SALARIO'].median()))
```


```python
# Vamos confirmar se não restaram valores nulos
df_dados.isnull().sum()
```




    UF                            0
    IDADE                         0
    ESCOLARIDADE                  0
    ESTADO_CIVIL                  0
    QT_FILHOS                     0
    CASA_PROPRIA                  0
    QT_IMOVEIS                    0
    VL_IMOVEIS                    0
    OUTRA_RENDA                   0
    OUTRA_RENDA_VALOR             0
    TEMPO_ULTIMO_EMPREGO_MESES    0
    TRABALHANDO_ATUALMENTE        0
    ULTIMO_SALARIO                0
    QT_CARROS                     0
    VALOR_TABELA_CARROS           0
    SCORE                         0
    dtype: int64




```python
# Vamos avaliar novamente os tipos das variaveis
df_dados.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10476 entries, 0 to 10475
    Data columns (total 16 columns):
     #   Column                      Non-Null Count  Dtype  
    ---  ------                      --------------  -----  
     0   UF                          10476 non-null  object 
     1   IDADE                       10476 non-null  int64  
     2   ESCOLARIDADE                10476 non-null  object 
     3   ESTADO_CIVIL                10476 non-null  object 
     4   QT_FILHOS                   10476 non-null  int64  
     5   CASA_PROPRIA                10476 non-null  object 
     6   QT_IMOVEIS                  10476 non-null  int64  
     7   VL_IMOVEIS                  10476 non-null  int64  
     8   OUTRA_RENDA                 10476 non-null  object 
     9   OUTRA_RENDA_VALOR           10476 non-null  int64  
     10  TEMPO_ULTIMO_EMPREGO_MESES  10476 non-null  int64  
     11  TRABALHANDO_ATUALMENTE      10476 non-null  object 
     12  ULTIMO_SALARIO              10476 non-null  float64
     13  QT_CARROS                   10476 non-null  int64  
     14  VALOR_TABELA_CARROS         10476 non-null  int64  
     15  SCORE                       10476 non-null  float64
    dtypes: float64(2), int64(8), object(6)
    memory usage: 1.3+ MB



```python
# Vamos avaliar algumas medidas estatisticas básicas
df_dados.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IDADE</th>
      <th>QT_FILHOS</th>
      <th>QT_IMOVEIS</th>
      <th>VL_IMOVEIS</th>
      <th>OUTRA_RENDA_VALOR</th>
      <th>TEMPO_ULTIMO_EMPREGO_MESES</th>
      <th>ULTIMO_SALARIO</th>
      <th>QT_CARROS</th>
      <th>VALOR_TABELA_CARROS</th>
      <th>SCORE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10476.000000</td>
      <td>10476.000000</td>
      <td>10476.000000</td>
      <td>10476.000000</td>
      <td>10476.000000</td>
      <td>10476.000000</td>
      <td>10476.000000</td>
      <td>10476.000000</td>
      <td>10476.000000</td>
      <td>10476.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>41.054124</td>
      <td>1.122566</td>
      <td>0.847079</td>
      <td>238453.608247</td>
      <td>641.237113</td>
      <td>43.070447</td>
      <td>8286.531119</td>
      <td>0.936426</td>
      <td>40996.563574</td>
      <td>51.058706</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.878162</td>
      <td>1.113537</td>
      <td>0.957374</td>
      <td>265843.934416</td>
      <td>1295.978195</td>
      <td>40.851521</td>
      <td>5826.589775</td>
      <td>0.806635</td>
      <td>47404.214062</td>
      <td>27.306340</td>
    </tr>
    <tr>
      <th>min</th>
      <td>19.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>1800.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.000000</td>
      <td>3900.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>28.666667</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>42.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>185000.000000</td>
      <td>0.000000</td>
      <td>22.000000</td>
      <td>6100.000000</td>
      <td>1.000000</td>
      <td>35000.000000</td>
      <td>45.166667</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>53.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>370000.000000</td>
      <td>0.000000</td>
      <td>75.000000</td>
      <td>11500.000000</td>
      <td>2.000000</td>
      <td>50000.000000</td>
      <td>72.666667</td>
    </tr>
    <tr>
      <th>max</th>
      <td>65.000000</td>
      <td>42.000000</td>
      <td>3.000000</td>
      <td>900000.000000</td>
      <td>4000.000000</td>
      <td>150.000000</td>
      <td>22000.000000</td>
      <td>2.000000</td>
      <td>180000.000000</td>
      <td>98.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Agora iremos avaliar os outliers das colunas que são númericas
# OUTLIERS são valores discrepantes que estão bem acima ou bem abaixo dos outros valores

# Vamos carregar em uma lista as variaveis que são do tipo INT64 E FLOAT64
variaveis_numericas = []
for i in df_dados.columns[0:16].tolist():
        if df_dados.dtypes[i] == 'int64' or df_dados.dtypes[i] == 'float64':            
            print(i, ':' , df_dados.dtypes[i]) 
            variaveis_numericas.append(i)
```

    IDADE : int64
    QT_FILHOS : int64
    QT_IMOVEIS : int64
    VL_IMOVEIS : int64
    OUTRA_RENDA_VALOR : int64
    TEMPO_ULTIMO_EMPREGO_MESES : int64
    ULTIMO_SALARIO : float64
    QT_CARROS : int64
    VALOR_TABELA_CARROS : int64
    SCORE : float64



```python
# Vamos observar a lista de variáveis e avaliar se nestas variáveis temos outliers através de um boxplot
variaveis_numericas
```




    ['IDADE',
     'QT_FILHOS',
     'QT_IMOVEIS',
     'VL_IMOVEIS',
     'OUTRA_RENDA_VALOR',
     'TEMPO_ULTIMO_EMPREGO_MESES',
     'ULTIMO_SALARIO',
     'QT_CARROS',
     'VALOR_TABELA_CARROS',
     'SCORE']




```python
# Com este comando iremos exibir todos gráficos de todas colunas de uma vez só para facilitar nossa analise.

# Aqui definimos o tamanho da tela para exibição dos gráficos
plt.rcParams["figure.figsize"] = [15.00, 12.00]
plt.rcParams["figure.autolayout"] = True

# Aqui definimos em quantas linhas e colunas queremos exibir os gráficos
f, axes = plt.subplots(2, 5) #2 linhas e 5 colunas

linha = 0
coluna = 0

for i in variaveis_numericas:
    sns.boxplot(data = df_dados, y=i, ax=axes[linha][coluna])
    coluna += 1
    if coluna == 5:
        linha += 1
        coluna = 0            

plt.show()
```


    
![png](ScoreCredito_files/ScoreCredito_20_0.png)
    



```python
# Agora já sabemos que temos possíveis OUTLIERS nas variáveis QT_FILHOS, QT_IMOVEIS, VALOR_TABELA_CARROS e OUTRA_RENDA_VALOR 
# Vamos olhar quais são esses outliers para avaliar como iremos trata-los.

# Vamos listar a quantidade de filhos superiores a 4
# Como temos somente 2 registros que realmente são outliers então iremos exclui-los
df_dados.loc[df_dados['QT_FILHOS'] > 4]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UF</th>
      <th>IDADE</th>
      <th>ESCOLARIDADE</th>
      <th>ESTADO_CIVIL</th>
      <th>QT_FILHOS</th>
      <th>CASA_PROPRIA</th>
      <th>QT_IMOVEIS</th>
      <th>VL_IMOVEIS</th>
      <th>OUTRA_RENDA</th>
      <th>OUTRA_RENDA_VALOR</th>
      <th>TEMPO_ULTIMO_EMPREGO_MESES</th>
      <th>TRABALHANDO_ATUALMENTE</th>
      <th>ULTIMO_SALARIO</th>
      <th>QT_CARROS</th>
      <th>VALOR_TABELA_CARROS</th>
      <th>SCORE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>SP</td>
      <td>48</td>
      <td>Superior Completo</td>
      <td>Divorciado</td>
      <td>38</td>
      <td>Sim</td>
      <td>2</td>
      <td>600000</td>
      <td>Não</td>
      <td>0</td>
      <td>15</td>
      <td>Sim</td>
      <td>15000.0</td>
      <td>1</td>
      <td>70000</td>
      <td>67.166667</td>
    </tr>
    <tr>
      <th>10455</th>
      <td>SP</td>
      <td>45</td>
      <td>Segundo Grau Completo</td>
      <td>Casado</td>
      <td>42</td>
      <td>Sim</td>
      <td>1</td>
      <td>220000</td>
      <td>Não</td>
      <td>0</td>
      <td>18</td>
      <td>Sim</td>
      <td>2200.0</td>
      <td>2</td>
      <td>30000</td>
      <td>23.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Nesse exemplo vamos excluir o registro todo.
df_dados.drop(df_dados.loc[df_dados['QT_FILHOS'] > 4].index, inplace=True)
```


```python
# Vamos avaliar as outras variaveis com possíveis outliers
```


```python
# Não precisamos alterar nada
df_dados.groupby(['OUTRA_RENDA_VALOR']).size()
```




    OUTRA_RENDA_VALOR
    0       8350
    2400     468
    3000     612
    3200     522
    4000     522
    dtype: int64




```python
# Não precisamos alterar nada
df_dados.groupby(['VALOR_TABELA_CARROS']).size()
```




    VALOR_TABELA_CARROS
    0         3762
    28000      468
    30000      791
    35000      792
    40000      792
    48000      522
    50000     1314
    70000      521
    80000      522
    150000     468
    180000     522
    dtype: int64




```python
# Não precisamos alterar nada
df_dados.groupby(['QT_IMOVEIS']).size()
```




    QT_IMOVEIS
    0    4680
    1    3761
    2     989
    3    1044
    dtype: int64




```python
# Vamos gerar um gráfico de histograma para avaliar a distribuição dos dados
# Podemos observar que neste caso os dados estão bem dispersos

# Aqui definimos o tamanho da tela para exibição dos gráficos
plt.rcParams["figure.figsize"] = [15.00, 12.00]
plt.rcParams["figure.autolayout"] = True

# Aqui definimos em quantas linhas e colunas queremos exibir os gráficos
f, axes = plt.subplots(4, 3) #4 linhas e 3 colunas

linha = 0
coluna = 0

for i in variaveis_numericas:
    sns.histplot(data = df_dados, x=i, ax=axes[linha][coluna])    
    coluna += 1
    if coluna == 3:
        linha += 1
        coluna = 0            

plt.show()
```


    
![png](ScoreCredito_files/ScoreCredito_27_0.png)
    



```python
# Vamos observar um grafico de dispersão para avaliar a correlação de algumas variaveis
sns.lmplot(x = "VL_IMOVEIS", y = "SCORE", data = df_dados);
```


    
![png](ScoreCredito_files/ScoreCredito_28_0.png)
    



```python
# Vamos observar um grafico de dispersão para avaliar a correlação de algumas variaveis
sns.lmplot(x = "ULTIMO_SALARIO", y = "SCORE", data = df_dados);
```


    
![png](ScoreCredito_files/ScoreCredito_29_0.png)
    



```python
# Vamos observar um grafico de dispersão para avaliar a correlação de algumas variaveis
sns.lmplot(x = "TEMPO_ULTIMO_EMPREGO_MESES", y = "SCORE", data = df_dados);
```


    
![png](ScoreCredito_files/ScoreCredito_30_0.png)
    



```python
# Vamos fazer uma engenharia de atributos no campo de IDADE e criar um novo campo de Faixa Etaria
print('Menor Idade: ', df_dados['IDADE'].min())
print('Maior Idade: ', df_dados['IDADE'].max())
```

    Menor Idade:  19
    Maior Idade:  65



```python
# Engenharia de Atributos - Iremos criar uma nova variável
idade_bins = [0, 30, 40, 50, 60]
idade_categoria = ["Até 30", "31 a 40", "41 a 50", "Maior que 50"]

df_dados["FAIXA_ETARIA"] = pd.cut(df_dados["IDADE"], idade_bins, labels=idade_categoria)

df_dados["FAIXA_ETARIA"].value_counts()
```




    FAIXA_ETARIA
    Até 30          3552
    Maior que 50    2448
    41 a 50         2070
    31 a 40         1270
    Name: count, dtype: int64




```python
# Vamos avaliar a média do score pela faixa etaria
df_dados.groupby(["FAIXA_ETARIA"]).mean(numeric_only=True)["SCORE"]
```




    FAIXA_ETARIA
    Até 30          44.762950
    31 a 40         48.883202
    41 a 50         51.440177
    Maior que 50    56.123775
    Name: SCORE, dtype: float64




```python
variaveis_categoricas = []
for i in df_dados.columns[0:48].tolist():
        if df_dados.dtypes[i] == 'object' or df_dados.dtypes[i] == 'category':            
            print(i, ':' , df_dados.dtypes[i]) 
            variaveis_categoricas.append(i)           
```

    UF : object
    ESCOLARIDADE : object
    ESTADO_CIVIL : object
    CASA_PROPRIA : object
    OUTRA_RENDA : object
    TRABALHANDO_ATUALMENTE : object
    FAIXA_ETARIA : category



```python
# Com este comando iremos exibir todos gráficos de todas colunas de uma vez só para facilitar nossa analise.

# Aqui definimos o tamanho da tela para exibição dos gráficos
plt.rcParams["figure.figsize"] = [15.00, 22.00]
plt.rcParams["figure.autolayout"] = True

# Aqui definimos em quantas linhas e colunas queremos exibir os gráficos
f, axes = plt.subplots(4, 2) #3 linhas e 2 colunas

linha = 0
coluna = 0

for i in variaveis_categoricas:    
    sns.countplot(data = df_dados, x=i, ax=axes[linha][coluna])
    
    coluna += 1
    if coluna == 2:
        linha += 1
        coluna = 0            

plt.show()
```


    
![png](ScoreCredito_files/ScoreCredito_35_0.png)
    


## Pré Processamento dos Dados


```python
# Cria o encoder
lb = LabelEncoder()

# Aplica o encoder nas variáveis que estão com string
df_dados['FAIXA_ETARIA'] = lb.fit_transform(df_dados['FAIXA_ETARIA'])
df_dados['OUTRA_RENDA'] = lb.fit_transform(df_dados['OUTRA_RENDA'])
df_dados['TRABALHANDO_ATUALMENTE'] = lb.fit_transform(df_dados['TRABALHANDO_ATUALMENTE'])
df_dados['ESTADO_CIVIL'] = lb.fit_transform(df_dados['ESTADO_CIVIL'])
df_dados['CASA_PROPRIA'] = lb.fit_transform(df_dados['CASA_PROPRIA'])
df_dados['ESCOLARIDADE'] = lb.fit_transform(df_dados['ESCOLARIDADE'])
df_dados['UF'] = lb.fit_transform(df_dados['UF'])

# Remove valores missing eventualmente gerados
df_dados.dropna(inplace = True)
```


```python
df_dados.head(200)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UF</th>
      <th>IDADE</th>
      <th>ESCOLARIDADE</th>
      <th>ESTADO_CIVIL</th>
      <th>QT_FILHOS</th>
      <th>CASA_PROPRIA</th>
      <th>QT_IMOVEIS</th>
      <th>VL_IMOVEIS</th>
      <th>OUTRA_RENDA</th>
      <th>OUTRA_RENDA_VALOR</th>
      <th>TEMPO_ULTIMO_EMPREGO_MESES</th>
      <th>TRABALHANDO_ATUALMENTE</th>
      <th>ULTIMO_SALARIO</th>
      <th>QT_CARROS</th>
      <th>VALOR_TABELA_CARROS</th>
      <th>SCORE</th>
      <th>FAIXA_ETARIA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>19</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>1800.0</td>
      <td>0</td>
      <td>0</td>
      <td>12.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>23</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>4800.0</td>
      <td>1</td>
      <td>50000</td>
      <td>18.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>220000</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>2200.0</td>
      <td>2</td>
      <td>30000</td>
      <td>23.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>27</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>0</td>
      <td>3900.0</td>
      <td>0</td>
      <td>0</td>
      <td>28.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>370000</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>6100.0</td>
      <td>1</td>
      <td>35000</td>
      <td>34.166667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>1</td>
      <td>3100.0</td>
      <td>2</td>
      <td>40000</td>
      <td>39.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>32</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>185000</td>
      <td>1</td>
      <td>3000</td>
      <td>19</td>
      <td>1</td>
      <td>6800.0</td>
      <td>0</td>
      <td>0</td>
      <td>45.166667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3</td>
      <td>35</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>450000</td>
      <td>1</td>
      <td>2400</td>
      <td>25</td>
      <td>1</td>
      <td>22000.0</td>
      <td>1</td>
      <td>150000</td>
      <td>50.666667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>1</td>
      <td>4500.0</td>
      <td>2</td>
      <td>28000</td>
      <td>56.166667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>45</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>9800.0</td>
      <td>0</td>
      <td>0</td>
      <td>61.666667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4</td>
      <td>48</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>600000</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>1</td>
      <td>15000.0</td>
      <td>1</td>
      <td>70000</td>
      <td>67.166667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>51</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>80</td>
      <td>0</td>
      <td>6100.0</td>
      <td>2</td>
      <td>48000</td>
      <td>72.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3</td>
      <td>53</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>280000</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>9000.0</td>
      <td>0</td>
      <td>0</td>
      <td>78.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>55</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>700000</td>
      <td>0</td>
      <td>0</td>
      <td>75</td>
      <td>0</td>
      <td>17500.0</td>
      <td>1</td>
      <td>50000</td>
      <td>83.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>340000</td>
      <td>1</td>
      <td>4000</td>
      <td>90</td>
      <td>1</td>
      <td>13000.0</td>
      <td>2</td>
      <td>180000</td>
      <td>89.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>15</th>
      <td>4</td>
      <td>62</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>450000</td>
      <td>1</td>
      <td>3200</td>
      <td>93</td>
      <td>0</td>
      <td>5300.0</td>
      <td>0</td>
      <td>0</td>
      <td>94.666667</td>
      <td>4</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2</td>
      <td>65</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>900000</td>
      <td>0</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>18300.0</td>
      <td>1</td>
      <td>80000</td>
      <td>98.000000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>17</th>
      <td>4</td>
      <td>19</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>1800.0</td>
      <td>0</td>
      <td>0</td>
      <td>12.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>23</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>4800.0</td>
      <td>1</td>
      <td>50000</td>
      <td>18.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>220000</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>2200.0</td>
      <td>2</td>
      <td>30000</td>
      <td>23.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>27</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>0</td>
      <td>3900.0</td>
      <td>0</td>
      <td>0</td>
      <td>28.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>370000</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>11500.0</td>
      <td>1</td>
      <td>35000</td>
      <td>34.166667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>4</td>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>1</td>
      <td>3100.0</td>
      <td>2</td>
      <td>40000</td>
      <td>39.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
      <td>32</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>185000</td>
      <td>1</td>
      <td>3000</td>
      <td>19</td>
      <td>1</td>
      <td>6800.0</td>
      <td>0</td>
      <td>0</td>
      <td>45.166667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3</td>
      <td>35</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>450000</td>
      <td>1</td>
      <td>2400</td>
      <td>25</td>
      <td>1</td>
      <td>22000.0</td>
      <td>1</td>
      <td>150000</td>
      <td>50.666667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>1</td>
      <td>4500.0</td>
      <td>2</td>
      <td>28000</td>
      <td>56.166667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2</td>
      <td>45</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>9800.0</td>
      <td>0</td>
      <td>0</td>
      <td>61.666667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
      <td>51</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>80</td>
      <td>0</td>
      <td>6100.0</td>
      <td>2</td>
      <td>48000</td>
      <td>72.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>29</th>
      <td>3</td>
      <td>53</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>280000</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>9000.0</td>
      <td>0</td>
      <td>0</td>
      <td>78.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1</td>
      <td>55</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>700000</td>
      <td>0</td>
      <td>0</td>
      <td>75</td>
      <td>0</td>
      <td>17500.0</td>
      <td>1</td>
      <td>50000</td>
      <td>83.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>340000</td>
      <td>1</td>
      <td>4000</td>
      <td>90</td>
      <td>1</td>
      <td>13000.0</td>
      <td>2</td>
      <td>180000</td>
      <td>89.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>32</th>
      <td>4</td>
      <td>62</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>450000</td>
      <td>1</td>
      <td>3200</td>
      <td>93</td>
      <td>0</td>
      <td>5300.0</td>
      <td>0</td>
      <td>0</td>
      <td>94.666667</td>
      <td>4</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2</td>
      <td>65</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>900000</td>
      <td>0</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>18300.0</td>
      <td>1</td>
      <td>80000</td>
      <td>98.000000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>34</th>
      <td>4</td>
      <td>19</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>1800.0</td>
      <td>0</td>
      <td>0</td>
      <td>12.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0</td>
      <td>23</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>4800.0</td>
      <td>1</td>
      <td>50000</td>
      <td>18.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>36</th>
      <td>3</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>220000</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>2200.0</td>
      <td>2</td>
      <td>30000</td>
      <td>23.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1</td>
      <td>27</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>0</td>
      <td>3900.0</td>
      <td>0</td>
      <td>0</td>
      <td>28.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>370000</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>11500.0</td>
      <td>1</td>
      <td>35000</td>
      <td>34.166667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>39</th>
      <td>4</td>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>1</td>
      <td>3100.0</td>
      <td>2</td>
      <td>40000</td>
      <td>39.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0</td>
      <td>32</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>185000</td>
      <td>1</td>
      <td>3000</td>
      <td>19</td>
      <td>1</td>
      <td>6800.0</td>
      <td>0</td>
      <td>0</td>
      <td>45.166667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>3</td>
      <td>35</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>450000</td>
      <td>1</td>
      <td>2400</td>
      <td>25</td>
      <td>1</td>
      <td>22000.0</td>
      <td>1</td>
      <td>150000</td>
      <td>50.666667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>1</td>
      <td>4500.0</td>
      <td>2</td>
      <td>28000</td>
      <td>56.166667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2</td>
      <td>45</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>9800.0</td>
      <td>0</td>
      <td>0</td>
      <td>61.666667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>44</th>
      <td>4</td>
      <td>48</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>600000</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>1</td>
      <td>15000.0</td>
      <td>1</td>
      <td>70000</td>
      <td>67.166667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0</td>
      <td>51</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>80</td>
      <td>0</td>
      <td>6100.0</td>
      <td>2</td>
      <td>48000</td>
      <td>72.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>46</th>
      <td>3</td>
      <td>53</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>280000</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>9000.0</td>
      <td>0</td>
      <td>0</td>
      <td>78.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1</td>
      <td>55</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>700000</td>
      <td>0</td>
      <td>0</td>
      <td>75</td>
      <td>0</td>
      <td>17500.0</td>
      <td>1</td>
      <td>50000</td>
      <td>83.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>48</th>
      <td>2</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>340000</td>
      <td>1</td>
      <td>4000</td>
      <td>90</td>
      <td>1</td>
      <td>13000.0</td>
      <td>2</td>
      <td>180000</td>
      <td>89.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>49</th>
      <td>4</td>
      <td>62</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>450000</td>
      <td>1</td>
      <td>3200</td>
      <td>93</td>
      <td>0</td>
      <td>5300.0</td>
      <td>0</td>
      <td>0</td>
      <td>94.666667</td>
      <td>4</td>
    </tr>
    <tr>
      <th>50</th>
      <td>2</td>
      <td>65</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>900000</td>
      <td>0</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>18300.0</td>
      <td>1</td>
      <td>80000</td>
      <td>98.000000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>51</th>
      <td>4</td>
      <td>19</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>1800.0</td>
      <td>0</td>
      <td>0</td>
      <td>12.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>52</th>
      <td>0</td>
      <td>23</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>4800.0</td>
      <td>1</td>
      <td>50000</td>
      <td>18.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>53</th>
      <td>3</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>220000</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>2200.0</td>
      <td>2</td>
      <td>30000</td>
      <td>23.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>54</th>
      <td>1</td>
      <td>27</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>0</td>
      <td>3900.0</td>
      <td>0</td>
      <td>0</td>
      <td>28.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>55</th>
      <td>2</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>370000</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>11500.0</td>
      <td>1</td>
      <td>35000</td>
      <td>34.166667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>56</th>
      <td>4</td>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>1</td>
      <td>3100.0</td>
      <td>2</td>
      <td>40000</td>
      <td>39.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>57</th>
      <td>0</td>
      <td>32</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>185000</td>
      <td>1</td>
      <td>3000</td>
      <td>19</td>
      <td>1</td>
      <td>6800.0</td>
      <td>0</td>
      <td>0</td>
      <td>45.166667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>58</th>
      <td>3</td>
      <td>35</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>450000</td>
      <td>1</td>
      <td>2400</td>
      <td>25</td>
      <td>1</td>
      <td>22000.0</td>
      <td>1</td>
      <td>150000</td>
      <td>50.666667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>59</th>
      <td>1</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>1</td>
      <td>4500.0</td>
      <td>2</td>
      <td>28000</td>
      <td>56.166667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>60</th>
      <td>2</td>
      <td>45</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>9800.0</td>
      <td>0</td>
      <td>0</td>
      <td>61.666667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>61</th>
      <td>4</td>
      <td>48</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>600000</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>1</td>
      <td>15000.0</td>
      <td>1</td>
      <td>70000</td>
      <td>67.166667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>62</th>
      <td>0</td>
      <td>51</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>80</td>
      <td>0</td>
      <td>6100.0</td>
      <td>2</td>
      <td>48000</td>
      <td>72.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>63</th>
      <td>3</td>
      <td>53</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>280000</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>9000.0</td>
      <td>0</td>
      <td>0</td>
      <td>78.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>64</th>
      <td>1</td>
      <td>55</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>700000</td>
      <td>0</td>
      <td>0</td>
      <td>75</td>
      <td>0</td>
      <td>17500.0</td>
      <td>1</td>
      <td>50000</td>
      <td>83.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>65</th>
      <td>2</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>340000</td>
      <td>1</td>
      <td>4000</td>
      <td>90</td>
      <td>1</td>
      <td>13000.0</td>
      <td>2</td>
      <td>180000</td>
      <td>89.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>66</th>
      <td>4</td>
      <td>62</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>450000</td>
      <td>1</td>
      <td>3200</td>
      <td>93</td>
      <td>0</td>
      <td>5300.0</td>
      <td>0</td>
      <td>0</td>
      <td>94.666667</td>
      <td>4</td>
    </tr>
    <tr>
      <th>67</th>
      <td>2</td>
      <td>65</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>900000</td>
      <td>0</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>18300.0</td>
      <td>1</td>
      <td>80000</td>
      <td>98.000000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>68</th>
      <td>4</td>
      <td>19</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>1800.0</td>
      <td>0</td>
      <td>0</td>
      <td>12.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>69</th>
      <td>0</td>
      <td>23</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>4800.0</td>
      <td>1</td>
      <td>50000</td>
      <td>18.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>70</th>
      <td>3</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>220000</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>2200.0</td>
      <td>2</td>
      <td>30000</td>
      <td>23.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>71</th>
      <td>1</td>
      <td>27</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>0</td>
      <td>3900.0</td>
      <td>0</td>
      <td>0</td>
      <td>28.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>72</th>
      <td>2</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>370000</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>11500.0</td>
      <td>1</td>
      <td>35000</td>
      <td>34.166667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>73</th>
      <td>4</td>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>1</td>
      <td>3100.0</td>
      <td>2</td>
      <td>40000</td>
      <td>39.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>74</th>
      <td>0</td>
      <td>32</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>185000</td>
      <td>1</td>
      <td>3000</td>
      <td>19</td>
      <td>1</td>
      <td>6800.0</td>
      <td>0</td>
      <td>0</td>
      <td>45.166667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>75</th>
      <td>3</td>
      <td>35</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>450000</td>
      <td>1</td>
      <td>2400</td>
      <td>25</td>
      <td>1</td>
      <td>22000.0</td>
      <td>1</td>
      <td>150000</td>
      <td>50.666667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>76</th>
      <td>1</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>1</td>
      <td>4500.0</td>
      <td>2</td>
      <td>28000</td>
      <td>56.166667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>77</th>
      <td>2</td>
      <td>45</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>9800.0</td>
      <td>0</td>
      <td>0</td>
      <td>61.666667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>78</th>
      <td>4</td>
      <td>48</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>600000</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>1</td>
      <td>15000.0</td>
      <td>1</td>
      <td>70000</td>
      <td>67.166667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0</td>
      <td>51</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>80</td>
      <td>0</td>
      <td>6100.0</td>
      <td>2</td>
      <td>48000</td>
      <td>72.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>80</th>
      <td>3</td>
      <td>53</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>280000</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>9000.0</td>
      <td>0</td>
      <td>0</td>
      <td>78.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>81</th>
      <td>1</td>
      <td>55</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>700000</td>
      <td>0</td>
      <td>0</td>
      <td>75</td>
      <td>0</td>
      <td>17500.0</td>
      <td>1</td>
      <td>50000</td>
      <td>83.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>340000</td>
      <td>1</td>
      <td>4000</td>
      <td>90</td>
      <td>1</td>
      <td>13000.0</td>
      <td>2</td>
      <td>180000</td>
      <td>89.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>83</th>
      <td>4</td>
      <td>62</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>450000</td>
      <td>1</td>
      <td>3200</td>
      <td>93</td>
      <td>0</td>
      <td>5300.0</td>
      <td>0</td>
      <td>0</td>
      <td>94.666667</td>
      <td>4</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2</td>
      <td>65</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>900000</td>
      <td>0</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>18300.0</td>
      <td>1</td>
      <td>80000</td>
      <td>98.000000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>85</th>
      <td>4</td>
      <td>19</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>1800.0</td>
      <td>0</td>
      <td>0</td>
      <td>12.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>86</th>
      <td>0</td>
      <td>23</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>4800.0</td>
      <td>1</td>
      <td>50000</td>
      <td>18.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>87</th>
      <td>3</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>220000</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>2200.0</td>
      <td>2</td>
      <td>30000</td>
      <td>23.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>88</th>
      <td>1</td>
      <td>27</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>0</td>
      <td>3900.0</td>
      <td>0</td>
      <td>0</td>
      <td>28.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>370000</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>11500.0</td>
      <td>1</td>
      <td>35000</td>
      <td>34.166667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>90</th>
      <td>4</td>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>1</td>
      <td>3100.0</td>
      <td>2</td>
      <td>40000</td>
      <td>39.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0</td>
      <td>32</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>185000</td>
      <td>1</td>
      <td>3000</td>
      <td>19</td>
      <td>1</td>
      <td>6800.0</td>
      <td>0</td>
      <td>0</td>
      <td>45.166667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>92</th>
      <td>3</td>
      <td>35</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>450000</td>
      <td>1</td>
      <td>2400</td>
      <td>25</td>
      <td>1</td>
      <td>22000.0</td>
      <td>1</td>
      <td>150000</td>
      <td>50.666667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>93</th>
      <td>1</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>1</td>
      <td>4500.0</td>
      <td>2</td>
      <td>28000</td>
      <td>56.166667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2</td>
      <td>45</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>9800.0</td>
      <td>0</td>
      <td>0</td>
      <td>61.666667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>95</th>
      <td>4</td>
      <td>48</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>600000</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>1</td>
      <td>15000.0</td>
      <td>1</td>
      <td>70000</td>
      <td>67.166667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0</td>
      <td>51</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>80</td>
      <td>0</td>
      <td>6100.0</td>
      <td>2</td>
      <td>48000</td>
      <td>72.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>97</th>
      <td>3</td>
      <td>53</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>280000</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>9000.0</td>
      <td>0</td>
      <td>0</td>
      <td>78.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>98</th>
      <td>1</td>
      <td>55</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>700000</td>
      <td>0</td>
      <td>0</td>
      <td>75</td>
      <td>0</td>
      <td>17500.0</td>
      <td>1</td>
      <td>50000</td>
      <td>83.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>340000</td>
      <td>1</td>
      <td>4000</td>
      <td>90</td>
      <td>1</td>
      <td>13000.0</td>
      <td>2</td>
      <td>180000</td>
      <td>89.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>100</th>
      <td>4</td>
      <td>62</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>450000</td>
      <td>1</td>
      <td>3200</td>
      <td>93</td>
      <td>0</td>
      <td>5300.0</td>
      <td>0</td>
      <td>0</td>
      <td>94.666667</td>
      <td>4</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2</td>
      <td>65</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>900000</td>
      <td>0</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>18300.0</td>
      <td>1</td>
      <td>80000</td>
      <td>98.000000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>102</th>
      <td>4</td>
      <td>19</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>1800.0</td>
      <td>0</td>
      <td>0</td>
      <td>12.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>103</th>
      <td>0</td>
      <td>23</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>4800.0</td>
      <td>1</td>
      <td>50000</td>
      <td>18.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>104</th>
      <td>3</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>220000</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>2200.0</td>
      <td>2</td>
      <td>30000</td>
      <td>23.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>105</th>
      <td>1</td>
      <td>27</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>0</td>
      <td>3900.0</td>
      <td>0</td>
      <td>0</td>
      <td>28.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>370000</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>11500.0</td>
      <td>1</td>
      <td>35000</td>
      <td>34.166667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>107</th>
      <td>4</td>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>1</td>
      <td>3100.0</td>
      <td>2</td>
      <td>40000</td>
      <td>39.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>108</th>
      <td>0</td>
      <td>32</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>185000</td>
      <td>1</td>
      <td>3000</td>
      <td>19</td>
      <td>1</td>
      <td>6800.0</td>
      <td>0</td>
      <td>0</td>
      <td>45.166667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>109</th>
      <td>3</td>
      <td>35</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>450000</td>
      <td>1</td>
      <td>2400</td>
      <td>25</td>
      <td>1</td>
      <td>22000.0</td>
      <td>1</td>
      <td>150000</td>
      <td>50.666667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>110</th>
      <td>1</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>1</td>
      <td>4500.0</td>
      <td>2</td>
      <td>28000</td>
      <td>56.166667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>111</th>
      <td>2</td>
      <td>45</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>9800.0</td>
      <td>0</td>
      <td>0</td>
      <td>61.666667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>112</th>
      <td>4</td>
      <td>48</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>600000</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>1</td>
      <td>15000.0</td>
      <td>1</td>
      <td>70000</td>
      <td>67.166667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>113</th>
      <td>0</td>
      <td>51</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>80</td>
      <td>0</td>
      <td>6100.0</td>
      <td>2</td>
      <td>48000</td>
      <td>72.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>114</th>
      <td>3</td>
      <td>53</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>280000</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>9000.0</td>
      <td>0</td>
      <td>0</td>
      <td>78.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>115</th>
      <td>1</td>
      <td>55</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>700000</td>
      <td>0</td>
      <td>0</td>
      <td>75</td>
      <td>0</td>
      <td>17500.0</td>
      <td>1</td>
      <td>50000</td>
      <td>83.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>116</th>
      <td>2</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>340000</td>
      <td>1</td>
      <td>4000</td>
      <td>90</td>
      <td>1</td>
      <td>13000.0</td>
      <td>2</td>
      <td>180000</td>
      <td>89.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>117</th>
      <td>4</td>
      <td>62</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>450000</td>
      <td>1</td>
      <td>3200</td>
      <td>93</td>
      <td>0</td>
      <td>5300.0</td>
      <td>0</td>
      <td>0</td>
      <td>94.666667</td>
      <td>4</td>
    </tr>
    <tr>
      <th>118</th>
      <td>2</td>
      <td>65</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>900000</td>
      <td>0</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>18300.0</td>
      <td>1</td>
      <td>80000</td>
      <td>98.000000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>119</th>
      <td>4</td>
      <td>19</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>1800.0</td>
      <td>0</td>
      <td>0</td>
      <td>12.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>120</th>
      <td>0</td>
      <td>23</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>4800.0</td>
      <td>1</td>
      <td>50000</td>
      <td>18.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>121</th>
      <td>3</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>220000</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>2200.0</td>
      <td>2</td>
      <td>30000</td>
      <td>23.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>122</th>
      <td>1</td>
      <td>27</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>0</td>
      <td>3900.0</td>
      <td>0</td>
      <td>0</td>
      <td>28.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>123</th>
      <td>2</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>370000</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>11500.0</td>
      <td>1</td>
      <td>35000</td>
      <td>34.166667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>124</th>
      <td>4</td>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>1</td>
      <td>3100.0</td>
      <td>2</td>
      <td>40000</td>
      <td>39.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>125</th>
      <td>0</td>
      <td>32</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>185000</td>
      <td>1</td>
      <td>3000</td>
      <td>19</td>
      <td>1</td>
      <td>6800.0</td>
      <td>0</td>
      <td>0</td>
      <td>45.166667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>126</th>
      <td>3</td>
      <td>35</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>450000</td>
      <td>1</td>
      <td>2400</td>
      <td>25</td>
      <td>1</td>
      <td>22000.0</td>
      <td>1</td>
      <td>150000</td>
      <td>50.666667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>127</th>
      <td>1</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>1</td>
      <td>4500.0</td>
      <td>2</td>
      <td>28000</td>
      <td>56.166667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>128</th>
      <td>2</td>
      <td>45</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>9800.0</td>
      <td>0</td>
      <td>0</td>
      <td>61.666667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>129</th>
      <td>4</td>
      <td>48</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>600000</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>1</td>
      <td>15000.0</td>
      <td>1</td>
      <td>70000</td>
      <td>67.166667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>130</th>
      <td>0</td>
      <td>51</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>80</td>
      <td>0</td>
      <td>6100.0</td>
      <td>2</td>
      <td>48000</td>
      <td>72.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>131</th>
      <td>3</td>
      <td>53</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>280000</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>9000.0</td>
      <td>0</td>
      <td>0</td>
      <td>78.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>132</th>
      <td>1</td>
      <td>55</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>700000</td>
      <td>0</td>
      <td>0</td>
      <td>75</td>
      <td>0</td>
      <td>17500.0</td>
      <td>1</td>
      <td>50000</td>
      <td>83.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>133</th>
      <td>2</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>340000</td>
      <td>1</td>
      <td>4000</td>
      <td>90</td>
      <td>1</td>
      <td>13000.0</td>
      <td>2</td>
      <td>180000</td>
      <td>89.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>134</th>
      <td>4</td>
      <td>62</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>450000</td>
      <td>1</td>
      <td>3200</td>
      <td>93</td>
      <td>0</td>
      <td>5300.0</td>
      <td>0</td>
      <td>0</td>
      <td>94.666667</td>
      <td>4</td>
    </tr>
    <tr>
      <th>135</th>
      <td>2</td>
      <td>65</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>900000</td>
      <td>0</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>18300.0</td>
      <td>1</td>
      <td>80000</td>
      <td>98.000000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>136</th>
      <td>4</td>
      <td>19</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>1800.0</td>
      <td>0</td>
      <td>0</td>
      <td>12.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>137</th>
      <td>0</td>
      <td>23</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>4800.0</td>
      <td>1</td>
      <td>50000</td>
      <td>18.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>138</th>
      <td>3</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>220000</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>2200.0</td>
      <td>2</td>
      <td>30000</td>
      <td>23.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>139</th>
      <td>1</td>
      <td>27</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>0</td>
      <td>3900.0</td>
      <td>0</td>
      <td>0</td>
      <td>28.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>140</th>
      <td>2</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>370000</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>11500.0</td>
      <td>1</td>
      <td>35000</td>
      <td>34.166667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>141</th>
      <td>4</td>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>1</td>
      <td>3100.0</td>
      <td>2</td>
      <td>40000</td>
      <td>39.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>142</th>
      <td>0</td>
      <td>32</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>185000</td>
      <td>1</td>
      <td>3000</td>
      <td>19</td>
      <td>1</td>
      <td>6800.0</td>
      <td>0</td>
      <td>0</td>
      <td>45.166667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>143</th>
      <td>3</td>
      <td>35</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>450000</td>
      <td>1</td>
      <td>2400</td>
      <td>25</td>
      <td>1</td>
      <td>22000.0</td>
      <td>1</td>
      <td>150000</td>
      <td>50.666667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>144</th>
      <td>1</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>1</td>
      <td>4500.0</td>
      <td>2</td>
      <td>28000</td>
      <td>56.166667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>145</th>
      <td>2</td>
      <td>45</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>9800.0</td>
      <td>0</td>
      <td>0</td>
      <td>61.666667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>146</th>
      <td>4</td>
      <td>48</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>600000</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>1</td>
      <td>15000.0</td>
      <td>1</td>
      <td>70000</td>
      <td>67.166667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>147</th>
      <td>0</td>
      <td>51</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>80</td>
      <td>0</td>
      <td>6100.0</td>
      <td>2</td>
      <td>48000</td>
      <td>72.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>148</th>
      <td>3</td>
      <td>53</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>280000</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>9000.0</td>
      <td>0</td>
      <td>0</td>
      <td>78.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>149</th>
      <td>1</td>
      <td>55</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>700000</td>
      <td>0</td>
      <td>0</td>
      <td>75</td>
      <td>0</td>
      <td>17500.0</td>
      <td>1</td>
      <td>50000</td>
      <td>83.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>150</th>
      <td>2</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>340000</td>
      <td>1</td>
      <td>4000</td>
      <td>90</td>
      <td>1</td>
      <td>13000.0</td>
      <td>2</td>
      <td>180000</td>
      <td>89.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>151</th>
      <td>4</td>
      <td>62</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>450000</td>
      <td>1</td>
      <td>3200</td>
      <td>93</td>
      <td>0</td>
      <td>5300.0</td>
      <td>0</td>
      <td>0</td>
      <td>94.666667</td>
      <td>4</td>
    </tr>
    <tr>
      <th>152</th>
      <td>2</td>
      <td>65</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>900000</td>
      <td>0</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>18300.0</td>
      <td>1</td>
      <td>80000</td>
      <td>98.000000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>153</th>
      <td>4</td>
      <td>19</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>1800.0</td>
      <td>0</td>
      <td>0</td>
      <td>12.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>154</th>
      <td>0</td>
      <td>23</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>4800.0</td>
      <td>1</td>
      <td>50000</td>
      <td>18.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>155</th>
      <td>3</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>220000</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>2200.0</td>
      <td>2</td>
      <td>30000</td>
      <td>23.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>156</th>
      <td>1</td>
      <td>27</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>0</td>
      <td>3900.0</td>
      <td>0</td>
      <td>0</td>
      <td>28.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>157</th>
      <td>2</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>370000</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>11500.0</td>
      <td>1</td>
      <td>35000</td>
      <td>34.166667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>158</th>
      <td>4</td>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>1</td>
      <td>3100.0</td>
      <td>2</td>
      <td>40000</td>
      <td>39.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>159</th>
      <td>0</td>
      <td>32</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>185000</td>
      <td>1</td>
      <td>3000</td>
      <td>19</td>
      <td>1</td>
      <td>6800.0</td>
      <td>0</td>
      <td>0</td>
      <td>45.166667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>160</th>
      <td>3</td>
      <td>35</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>1800.0</td>
      <td>0</td>
      <td>0</td>
      <td>12.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>161</th>
      <td>1</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>4800.0</td>
      <td>1</td>
      <td>50000</td>
      <td>18.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>162</th>
      <td>2</td>
      <td>45</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>220000</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>2200.0</td>
      <td>2</td>
      <td>30000</td>
      <td>23.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>163</th>
      <td>4</td>
      <td>48</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>0</td>
      <td>3900.0</td>
      <td>0</td>
      <td>0</td>
      <td>28.666667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>164</th>
      <td>0</td>
      <td>51</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>370000</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>11500.0</td>
      <td>1</td>
      <td>35000</td>
      <td>34.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>165</th>
      <td>3</td>
      <td>53</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>1</td>
      <td>3100.0</td>
      <td>2</td>
      <td>40000</td>
      <td>39.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>166</th>
      <td>1</td>
      <td>55</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>185000</td>
      <td>1</td>
      <td>3000</td>
      <td>19</td>
      <td>1</td>
      <td>6800.0</td>
      <td>0</td>
      <td>0</td>
      <td>45.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>167</th>
      <td>2</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>450000</td>
      <td>1</td>
      <td>2400</td>
      <td>25</td>
      <td>1</td>
      <td>22000.0</td>
      <td>1</td>
      <td>150000</td>
      <td>50.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>168</th>
      <td>4</td>
      <td>62</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>1</td>
      <td>4500.0</td>
      <td>2</td>
      <td>28000</td>
      <td>56.166667</td>
      <td>4</td>
    </tr>
    <tr>
      <th>169</th>
      <td>2</td>
      <td>65</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>9800.0</td>
      <td>0</td>
      <td>0</td>
      <td>61.666667</td>
      <td>4</td>
    </tr>
    <tr>
      <th>170</th>
      <td>4</td>
      <td>19</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>600000</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>1</td>
      <td>15000.0</td>
      <td>1</td>
      <td>70000</td>
      <td>67.166667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>171</th>
      <td>0</td>
      <td>23</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>80</td>
      <td>0</td>
      <td>6100.0</td>
      <td>2</td>
      <td>48000</td>
      <td>72.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>172</th>
      <td>3</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>280000</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>9000.0</td>
      <td>0</td>
      <td>0</td>
      <td>78.166667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>173</th>
      <td>1</td>
      <td>25</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>700000</td>
      <td>0</td>
      <td>0</td>
      <td>75</td>
      <td>0</td>
      <td>17500.0</td>
      <td>1</td>
      <td>50000</td>
      <td>83.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>174</th>
      <td>2</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>340000</td>
      <td>1</td>
      <td>4000</td>
      <td>90</td>
      <td>1</td>
      <td>13000.0</td>
      <td>2</td>
      <td>180000</td>
      <td>89.166667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>175</th>
      <td>4</td>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>450000</td>
      <td>1</td>
      <td>3200</td>
      <td>93</td>
      <td>0</td>
      <td>5300.0</td>
      <td>0</td>
      <td>0</td>
      <td>94.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>176</th>
      <td>0</td>
      <td>32</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>900000</td>
      <td>0</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>18300.0</td>
      <td>1</td>
      <td>80000</td>
      <td>98.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>177</th>
      <td>3</td>
      <td>35</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>1800.0</td>
      <td>0</td>
      <td>0</td>
      <td>12.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>178</th>
      <td>1</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>4800.0</td>
      <td>1</td>
      <td>50000</td>
      <td>18.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>179</th>
      <td>2</td>
      <td>45</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>220000</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>2200.0</td>
      <td>2</td>
      <td>30000</td>
      <td>23.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>180</th>
      <td>4</td>
      <td>48</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>0</td>
      <td>3900.0</td>
      <td>0</td>
      <td>0</td>
      <td>28.666667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>181</th>
      <td>0</td>
      <td>51</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>370000</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>11500.0</td>
      <td>1</td>
      <td>35000</td>
      <td>34.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>182</th>
      <td>3</td>
      <td>53</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>1</td>
      <td>3100.0</td>
      <td>2</td>
      <td>40000</td>
      <td>39.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>183</th>
      <td>1</td>
      <td>55</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>185000</td>
      <td>1</td>
      <td>3000</td>
      <td>19</td>
      <td>1</td>
      <td>6800.0</td>
      <td>0</td>
      <td>0</td>
      <td>45.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>184</th>
      <td>2</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>1800.0</td>
      <td>0</td>
      <td>0</td>
      <td>12.000000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>185</th>
      <td>4</td>
      <td>62</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>4800.0</td>
      <td>1</td>
      <td>50000</td>
      <td>18.000000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>186</th>
      <td>2</td>
      <td>65</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>220000</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>2200.0</td>
      <td>2</td>
      <td>30000</td>
      <td>23.000000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>187</th>
      <td>4</td>
      <td>19</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>0</td>
      <td>3900.0</td>
      <td>0</td>
      <td>0</td>
      <td>28.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>188</th>
      <td>0</td>
      <td>23</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>370000</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>11500.0</td>
      <td>1</td>
      <td>35000</td>
      <td>34.166667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>189</th>
      <td>3</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>1</td>
      <td>3100.0</td>
      <td>2</td>
      <td>40000</td>
      <td>39.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>190</th>
      <td>1</td>
      <td>27</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>185000</td>
      <td>1</td>
      <td>3000</td>
      <td>19</td>
      <td>1</td>
      <td>6800.0</td>
      <td>0</td>
      <td>0</td>
      <td>45.166667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>191</th>
      <td>2</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>450000</td>
      <td>1</td>
      <td>2400</td>
      <td>25</td>
      <td>1</td>
      <td>22000.0</td>
      <td>1</td>
      <td>150000</td>
      <td>50.666667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>192</th>
      <td>4</td>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>1</td>
      <td>4500.0</td>
      <td>2</td>
      <td>28000</td>
      <td>56.166667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>193</th>
      <td>0</td>
      <td>32</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>9800.0</td>
      <td>0</td>
      <td>0</td>
      <td>61.666667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194</th>
      <td>3</td>
      <td>35</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>600000</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>1</td>
      <td>15000.0</td>
      <td>1</td>
      <td>70000</td>
      <td>67.166667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>195</th>
      <td>1</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>80</td>
      <td>0</td>
      <td>6100.0</td>
      <td>2</td>
      <td>48000</td>
      <td>72.666667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>196</th>
      <td>2</td>
      <td>45</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>280000</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>9000.0</td>
      <td>0</td>
      <td>0</td>
      <td>78.166667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>197</th>
      <td>4</td>
      <td>48</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>700000</td>
      <td>0</td>
      <td>0</td>
      <td>75</td>
      <td>0</td>
      <td>17500.0</td>
      <td>1</td>
      <td>50000</td>
      <td>83.666667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>198</th>
      <td>0</td>
      <td>51</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>340000</td>
      <td>1</td>
      <td>4000</td>
      <td>90</td>
      <td>1</td>
      <td>13000.0</td>
      <td>2</td>
      <td>180000</td>
      <td>89.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>199</th>
      <td>3</td>
      <td>53</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>450000</td>
      <td>1</td>
      <td>3200</td>
      <td>93</td>
      <td>0</td>
      <td>5300.0</td>
      <td>0</td>
      <td>0</td>
      <td>94.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>200</th>
      <td>1</td>
      <td>55</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>900000</td>
      <td>0</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>18300.0</td>
      <td>1</td>
      <td>80000</td>
      <td>98.000000</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Agora podemos observar que já temos todas variaveis numericas
df_dados.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 10474 entries, 0 to 10475
    Data columns (total 17 columns):
     #   Column                      Non-Null Count  Dtype  
    ---  ------                      --------------  -----  
     0   UF                          10474 non-null  int64  
     1   IDADE                       10474 non-null  int64  
     2   ESCOLARIDADE                10474 non-null  int64  
     3   ESTADO_CIVIL                10474 non-null  int64  
     4   QT_FILHOS                   10474 non-null  int64  
     5   CASA_PROPRIA                10474 non-null  int64  
     6   QT_IMOVEIS                  10474 non-null  int64  
     7   VL_IMOVEIS                  10474 non-null  int64  
     8   OUTRA_RENDA                 10474 non-null  int64  
     9   OUTRA_RENDA_VALOR           10474 non-null  int64  
     10  TEMPO_ULTIMO_EMPREGO_MESES  10474 non-null  int64  
     11  TRABALHANDO_ATUALMENTE      10474 non-null  int64  
     12  ULTIMO_SALARIO              10474 non-null  float64
     13  QT_CARROS                   10474 non-null  int64  
     14  VALOR_TABELA_CARROS         10474 non-null  int64  
     15  SCORE                       10474 non-null  float64
     16  FAIXA_ETARIA                10474 non-null  int64  
    dtypes: float64(2), int64(15)
    memory usage: 1.4 MB



```python
# Separando a variavel alvo
target = df_dados.iloc[:,15:16]
```


```python
# Separando as variaveis preditoras

preditoras = df_dados.copy() #Fazendo uma cópia do dataframe

del preditoras['SCORE'] #Excluindo a variavel target, pois já separamos ela na etapa anterior

preditoras.head()#Visualizando as variaveis preditoras
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UF</th>
      <th>IDADE</th>
      <th>ESCOLARIDADE</th>
      <th>ESTADO_CIVIL</th>
      <th>QT_FILHOS</th>
      <th>CASA_PROPRIA</th>
      <th>QT_IMOVEIS</th>
      <th>VL_IMOVEIS</th>
      <th>OUTRA_RENDA</th>
      <th>OUTRA_RENDA_VALOR</th>
      <th>TEMPO_ULTIMO_EMPREGO_MESES</th>
      <th>TRABALHANDO_ATUALMENTE</th>
      <th>ULTIMO_SALARIO</th>
      <th>QT_CARROS</th>
      <th>VALOR_TABELA_CARROS</th>
      <th>FAIXA_ETARIA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>19</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>1800.0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>23</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>4800.0</td>
      <td>1</td>
      <td>50000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>220000</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>2200.0</td>
      <td>2</td>
      <td>30000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>27</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>0</td>
      <td>3900.0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>370000</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>6100.0</td>
      <td>1</td>
      <td>35000</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Divisão em Dados de Treino e Teste.
X_treino, X_teste, y_treino, y_teste = train_test_split(preditoras, target, test_size = 0.3, random_state = 40)
```


```python
# Vamos aplicar a normalização em treino e teste
# Padronização
sc = MinMaxScaler()
X_treino_normalizados = sc.fit_transform(X_treino)
X_teste_normalizados = sc.transform(X_teste)
```

## Criar, avaliar e testar nosso modelo preditivo 


```python
# Treina o modelo
modelo = LinearRegression()

modelo = modelo.fit(X_treino_normalizados, y_treino)
```


```python
r2_score(y_teste, modelo.fit(X_treino_normalizados, y_treino).predict(X_teste_normalizados))
```




    0.7984013631162861




```python
UF = 2
IDADE = 42 
ESCOLARIDADE = 1
ESTADO_CIVIL = 2
QT_FILHOS = 1
CASA_PROPRIA = 1
QT_IMOVEIS = 1
VL_IMOVEIS = 500000
OUTRA_RENDA = 1
OUTRA_RENDA_VALOR = 5000 
TEMPO_ULTIMO_EMPREGO_MESES = 18 
TRABALHANDO_ATUALMENTE = 1
ULTIMO_SALARIO = 5400.0
QT_CARROS = 4
VALOR_TABELA_CARROS = 70000
FAIXA_ETARIA = 3

novos_dados = [UF, IDADE, ESCOLARIDADE, ESTADO_CIVIL, QT_FILHOS,CASA_PROPRIA,QT_IMOVEIS,VL_IMOVEIS,OUTRA_RENDA,
               OUTRA_RENDA_VALOR,TEMPO_ULTIMO_EMPREGO_MESES,TRABALHANDO_ATUALMENTE,ULTIMO_SALARIO,QT_CARROS,
               VALOR_TABELA_CARROS, FAIXA_ETARIA]


# Reshape
X = np.array(novos_dados).reshape(1, -1)
X = sc.transform(X)

# Previsão
print("Score de crédito previsto para esse cliente:", modelo.predict(X))
```

    Score de crédito previsto para esse cliente: [[150.39122788]]



```python

```
