# Flats-cost-prediction

Подключаем необходимые библиотеки


```python
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```

Объявляем функции


```python
#Вычисляет среднюю абсолютную процентную ошибку
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#Вычисляет медианную абсолютную процентную ошибку
def median_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100

#Печатает рассчитанные значения коэффициента детерминации, средней и медианной абсолютных ошибок
def print_metrics(prediction, val_y):
    val_mae = mean_absolute_error(val_y, prediction)
    median_AE = median_absolute_error(val_y, prediction)
    r2 = r2_score(val_y, prediction)

    print('')
    print('R\u00b2: {:.2}'.format(r2))
    print('')
    print('Средняя абсолютная ошибка: {:.3} %'.format(mean_absolute_percentage_error(val_y, prediction)))
    print('Медианная абсолютная ошибка: {:.3} %'.format(median_absolute_percentage_error(val_y, prediction)))
```

Загружаем датасет и делаем первичную обработку


```python
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_df.fillna(0)
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
      <th>Id</th>
      <th>DistrictId</th>
      <th>Rooms</th>
      <th>Square</th>
      <th>LifeSquare</th>
      <th>KitchenSquare</th>
      <th>Floor</th>
      <th>HouseFloor</th>
      <th>HouseYear</th>
      <th>Ecology_1</th>
      <th>Ecology_2</th>
      <th>Ecology_3</th>
      <th>Social_1</th>
      <th>Social_2</th>
      <th>Social_3</th>
      <th>Healthcare_1</th>
      <th>Helthcare_2</th>
      <th>Shops_1</th>
      <th>Shops_2</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5613</td>
      <td>145</td>
      <td>1.0</td>
      <td>39.832524</td>
      <td>23.169223</td>
      <td>8.0</td>
      <td>7</td>
      <td>8.0</td>
      <td>1966</td>
      <td>0.118537</td>
      <td>B</td>
      <td>B</td>
      <td>30</td>
      <td>6207</td>
      <td>1</td>
      <td>1183.0</td>
      <td>1</td>
      <td>0</td>
      <td>B</td>
      <td>177734.553407</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7968</td>
      <td>143</td>
      <td>3.0</td>
      <td>78.342215</td>
      <td>47.671972</td>
      <td>10.0</td>
      <td>2</td>
      <td>17.0</td>
      <td>1988</td>
      <td>0.025609</td>
      <td>B</td>
      <td>B</td>
      <td>33</td>
      <td>5261</td>
      <td>0</td>
      <td>240.0</td>
      <td>3</td>
      <td>1</td>
      <td>B</td>
      <td>282078.720850</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7842</td>
      <td>143</td>
      <td>1.0</td>
      <td>40.409907</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>10</td>
      <td>22.0</td>
      <td>1977</td>
      <td>0.007122</td>
      <td>B</td>
      <td>B</td>
      <td>1</td>
      <td>264</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>B</td>
      <td>168106.007630</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1797</td>
      <td>7</td>
      <td>2.0</td>
      <td>64.285067</td>
      <td>38.562517</td>
      <td>9.0</td>
      <td>16</td>
      <td>16.0</td>
      <td>1972</td>
      <td>0.282798</td>
      <td>B</td>
      <td>B</td>
      <td>33</td>
      <td>8667</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>6</td>
      <td>B</td>
      <td>343995.102962</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8933</td>
      <td>171</td>
      <td>3.0</td>
      <td>62.528465</td>
      <td>47.103833</td>
      <td>6.0</td>
      <td>9</td>
      <td>9.0</td>
      <td>1972</td>
      <td>0.012339</td>
      <td>B</td>
      <td>B</td>
      <td>35</td>
      <td>5776</td>
      <td>1</td>
      <td>2078.0</td>
      <td>2</td>
      <td>4</td>
      <td>B</td>
      <td>161044.944138</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7995</th>
      <td>8203</td>
      <td>63</td>
      <td>1.0</td>
      <td>43.303458</td>
      <td>21.519087</td>
      <td>8.0</td>
      <td>12</td>
      <td>14.0</td>
      <td>1992</td>
      <td>0.161532</td>
      <td>B</td>
      <td>B</td>
      <td>25</td>
      <td>5648</td>
      <td>1</td>
      <td>30.0</td>
      <td>2</td>
      <td>4</td>
      <td>B</td>
      <td>136744.340827</td>
    </tr>
    <tr>
      <th>7996</th>
      <td>5013</td>
      <td>58</td>
      <td>2.0</td>
      <td>49.090728</td>
      <td>33.272626</td>
      <td>6.0</td>
      <td>3</td>
      <td>12.0</td>
      <td>1981</td>
      <td>0.300323</td>
      <td>B</td>
      <td>B</td>
      <td>52</td>
      <td>10311</td>
      <td>6</td>
      <td>0.0</td>
      <td>1</td>
      <td>9</td>
      <td>B</td>
      <td>119367.455796</td>
    </tr>
    <tr>
      <th>7997</th>
      <td>7685</td>
      <td>207</td>
      <td>2.0</td>
      <td>64.307684</td>
      <td>37.038420</td>
      <td>9.0</td>
      <td>13</td>
      <td>0.0</td>
      <td>1977</td>
      <td>0.072158</td>
      <td>B</td>
      <td>B</td>
      <td>2</td>
      <td>629</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>199715.148807</td>
    </tr>
    <tr>
      <th>7998</th>
      <td>6783</td>
      <td>44</td>
      <td>1.0</td>
      <td>29.648057</td>
      <td>16.555363</td>
      <td>5.0</td>
      <td>3</td>
      <td>5.0</td>
      <td>1958</td>
      <td>0.460556</td>
      <td>B</td>
      <td>B</td>
      <td>20</td>
      <td>4386</td>
      <td>14</td>
      <td>0.0</td>
      <td>1</td>
      <td>5</td>
      <td>B</td>
      <td>165953.912580</td>
    </tr>
    <tr>
      <th>7999</th>
      <td>8368</td>
      <td>141</td>
      <td>1.0</td>
      <td>32.330292</td>
      <td>22.326870</td>
      <td>5.0</td>
      <td>3</td>
      <td>9.0</td>
      <td>1969</td>
      <td>0.194489</td>
      <td>B</td>
      <td>B</td>
      <td>47</td>
      <td>8004</td>
      <td>3</td>
      <td>125.0</td>
      <td>3</td>
      <td>5</td>
      <td>B</td>
      <td>171842.411855</td>
    </tr>
  </tbody>
</table>
<p>8000 rows × 20 columns</p>
</div>




```python
test_df.fillna(0)
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
      <th>Id</th>
      <th>DistrictId</th>
      <th>Rooms</th>
      <th>Square</th>
      <th>LifeSquare</th>
      <th>KitchenSquare</th>
      <th>Floor</th>
      <th>HouseFloor</th>
      <th>HouseYear</th>
      <th>Ecology_1</th>
      <th>Ecology_2</th>
      <th>Ecology_3</th>
      <th>Social_1</th>
      <th>Social_2</th>
      <th>Social_3</th>
      <th>Healthcare_1</th>
      <th>Helthcare_2</th>
      <th>Shops_1</th>
      <th>Shops_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1109</td>
      <td>207</td>
      <td>3.0</td>
      <td>115.027311</td>
      <td>0.000000</td>
      <td>10.0</td>
      <td>4</td>
      <td>10.0</td>
      <td>2014</td>
      <td>0.075424</td>
      <td>B</td>
      <td>B</td>
      <td>11</td>
      <td>3097</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5304</td>
      <td>143</td>
      <td>1.0</td>
      <td>46.887892</td>
      <td>44.628132</td>
      <td>1.0</td>
      <td>12</td>
      <td>20.0</td>
      <td>1977</td>
      <td>0.007122</td>
      <td>B</td>
      <td>B</td>
      <td>1</td>
      <td>264</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7077</td>
      <td>26</td>
      <td>2.0</td>
      <td>53.975144</td>
      <td>34.153584</td>
      <td>8.0</td>
      <td>2</td>
      <td>12.0</td>
      <td>1978</td>
      <td>0.127376</td>
      <td>B</td>
      <td>B</td>
      <td>43</td>
      <td>8429</td>
      <td>3</td>
      <td>0.0</td>
      <td>3</td>
      <td>9</td>
      <td>B</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2199</td>
      <td>178</td>
      <td>1.0</td>
      <td>36.673407</td>
      <td>16.285522</td>
      <td>9.0</td>
      <td>3</td>
      <td>12.0</td>
      <td>2003</td>
      <td>0.041116</td>
      <td>B</td>
      <td>B</td>
      <td>53</td>
      <td>14892</td>
      <td>4</td>
      <td>0.0</td>
      <td>1</td>
      <td>4</td>
      <td>B</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5205</td>
      <td>174</td>
      <td>2.0</td>
      <td>46.760177</td>
      <td>29.829147</td>
      <td>6.0</td>
      <td>4</td>
      <td>9.0</td>
      <td>1974</td>
      <td>0.089040</td>
      <td>B</td>
      <td>B</td>
      <td>33</td>
      <td>7976</td>
      <td>5</td>
      <td>0.0</td>
      <td>0</td>
      <td>11</td>
      <td>B</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>7153</td>
      <td>21</td>
      <td>2.0</td>
      <td>55.860952</td>
      <td>34.465327</td>
      <td>8.0</td>
      <td>3</td>
      <td>7.0</td>
      <td>1961</td>
      <td>0.049863</td>
      <td>B</td>
      <td>B</td>
      <td>18</td>
      <td>3746</td>
      <td>9</td>
      <td>75.0</td>
      <td>3</td>
      <td>1</td>
      <td>B</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>7729</td>
      <td>101</td>
      <td>2.0</td>
      <td>38.676356</td>
      <td>22.427115</td>
      <td>6.0</td>
      <td>2</td>
      <td>9.0</td>
      <td>1967</td>
      <td>0.000000</td>
      <td>B</td>
      <td>B</td>
      <td>18</td>
      <td>3374</td>
      <td>5</td>
      <td>620.0</td>
      <td>1</td>
      <td>2</td>
      <td>B</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>6128</td>
      <td>81</td>
      <td>1.0</td>
      <td>34.723984</td>
      <td>19.840550</td>
      <td>9.0</td>
      <td>6</td>
      <td>16.0</td>
      <td>1988</td>
      <td>0.521867</td>
      <td>B</td>
      <td>B</td>
      <td>25</td>
      <td>6149</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>9764</td>
      <td>188</td>
      <td>2.0</td>
      <td>50.902724</td>
      <td>27.159548</td>
      <td>6.0</td>
      <td>4</td>
      <td>9.0</td>
      <td>1972</td>
      <td>0.127812</td>
      <td>B</td>
      <td>B</td>
      <td>28</td>
      <td>7287</td>
      <td>5</td>
      <td>320.0</td>
      <td>1</td>
      <td>3</td>
      <td>B</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>2569</td>
      <td>7</td>
      <td>1.0</td>
      <td>35.815476</td>
      <td>22.301367</td>
      <td>6.0</td>
      <td>9</td>
      <td>9.0</td>
      <td>1975</td>
      <td>0.127376</td>
      <td>B</td>
      <td>B</td>
      <td>43</td>
      <td>8429</td>
      <td>3</td>
      <td>0.0</td>
      <td>3</td>
      <td>9</td>
      <td>B</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 19 columns</p>
</div>




```python
train_df.describe()
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
      <th>Id</th>
      <th>DistrictId</th>
      <th>Rooms</th>
      <th>Square</th>
      <th>LifeSquare</th>
      <th>KitchenSquare</th>
      <th>Floor</th>
      <th>HouseFloor</th>
      <th>HouseYear</th>
      <th>Ecology_1</th>
      <th>Social_1</th>
      <th>Social_2</th>
      <th>Social_3</th>
      <th>Healthcare_1</th>
      <th>Helthcare_2</th>
      <th>Shops_1</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>6296.000000</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>8.000000e+03</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>4155.000000</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4987.608750</td>
      <td>110.822500</td>
      <td>1.893750</td>
      <td>56.366504</td>
      <td>37.348690</td>
      <td>6.106375</td>
      <td>8.541000</td>
      <td>12.598875</td>
      <td>4.491401e+03</td>
      <td>0.118622</td>
      <td>24.661375</td>
      <td>5332.776250</td>
      <td>7.847125</td>
      <td>1143.246209</td>
      <td>1.324000</td>
      <td>4.194125</td>
      <td>214605.477542</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2882.340831</td>
      <td>57.413814</td>
      <td>0.825861</td>
      <td>20.518022</td>
      <td>95.667433</td>
      <td>22.521905</td>
      <td>5.256118</td>
      <td>6.851795</td>
      <td>2.241661e+05</td>
      <td>0.119217</td>
      <td>17.512011</td>
      <td>3991.217955</td>
      <td>23.224156</td>
      <td>1033.605378</td>
      <td>1.497933</td>
      <td>4.735477</td>
      <td>93550.205075</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.136859</td>
      <td>0.370619</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.910000e+03</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>168.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>59174.778028</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2485.750000</td>
      <td>63.000000</td>
      <td>1.000000</td>
      <td>41.800063</td>
      <td>22.765329</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>9.000000</td>
      <td>1.974000e+03</td>
      <td>0.017647</td>
      <td>6.000000</td>
      <td>1564.000000</td>
      <td>0.000000</td>
      <td>325.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>153994.680334</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4987.500000</td>
      <td>116.000000</td>
      <td>2.000000</td>
      <td>52.619610</td>
      <td>32.726626</td>
      <td>6.000000</td>
      <td>7.000000</td>
      <td>12.000000</td>
      <td>1.977000e+03</td>
      <td>0.075424</td>
      <td>25.000000</td>
      <td>5285.000000</td>
      <td>2.000000</td>
      <td>900.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>192034.691671</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7470.500000</td>
      <td>145.000000</td>
      <td>2.000000</td>
      <td>66.036608</td>
      <td>45.204687</td>
      <td>9.000000</td>
      <td>12.000000</td>
      <td>17.000000</td>
      <td>2.001000e+03</td>
      <td>0.195781</td>
      <td>36.000000</td>
      <td>7227.000000</td>
      <td>5.000000</td>
      <td>1548.000000</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>249970.954618</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9999.000000</td>
      <td>209.000000</td>
      <td>10.000000</td>
      <td>604.705972</td>
      <td>7480.592129</td>
      <td>1970.000000</td>
      <td>42.000000</td>
      <td>117.000000</td>
      <td>2.005201e+07</td>
      <td>0.521867</td>
      <td>74.000000</td>
      <td>19083.000000</td>
      <td>141.000000</td>
      <td>4849.000000</td>
      <td>6.000000</td>
      <td>23.000000</td>
      <td>633233.466570</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df.describe()
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
      <th>Id</th>
      <th>DistrictId</th>
      <th>Rooms</th>
      <th>Square</th>
      <th>LifeSquare</th>
      <th>KitchenSquare</th>
      <th>Floor</th>
      <th>HouseFloor</th>
      <th>HouseYear</th>
      <th>Ecology_1</th>
      <th>Social_1</th>
      <th>Social_2</th>
      <th>Social_3</th>
      <th>Healthcare_1</th>
      <th>Helthcare_2</th>
      <th>Shops_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>1591.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>1047.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5047.065000</td>
      <td>109.961000</td>
      <td>1.877500</td>
      <td>56.112859</td>
      <td>36.609834</td>
      <td>6.941000</td>
      <td>8.469500</td>
      <td>12.651500</td>
      <td>1985.226000</td>
      <td>0.119802</td>
      <td>24.789500</td>
      <td>5429.682000</td>
      <td>8.807500</td>
      <td>1141.548233</td>
      <td>1.301500</td>
      <td>4.380000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2905.281024</td>
      <td>57.369993</td>
      <td>0.892128</td>
      <td>23.099741</td>
      <td>25.554637</td>
      <td>45.276909</td>
      <td>5.181746</td>
      <td>6.465288</td>
      <td>18.455249</td>
      <td>0.118281</td>
      <td>17.618808</td>
      <td>4068.616739</td>
      <td>26.113874</td>
      <td>972.540534</td>
      <td>1.476377</td>
      <td>5.078434</td>
    </tr>
    <tr>
      <th>min</th>
      <td>12.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.596351</td>
      <td>1.104689</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1912.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>168.000000</td>
      <td>0.000000</td>
      <td>30.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2559.250000</td>
      <td>63.000000</td>
      <td>1.000000</td>
      <td>41.637552</td>
      <td>22.818785</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>9.000000</td>
      <td>1974.000000</td>
      <td>0.020741</td>
      <td>6.000000</td>
      <td>1741.000000</td>
      <td>0.000000</td>
      <td>540.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5034.000000</td>
      <td>113.000000</td>
      <td>2.000000</td>
      <td>51.905879</td>
      <td>32.894101</td>
      <td>6.000000</td>
      <td>7.000000</td>
      <td>14.000000</td>
      <td>1977.000000</td>
      <td>0.075779</td>
      <td>25.000000</td>
      <td>5288.000000</td>
      <td>2.000000</td>
      <td>990.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7576.500000</td>
      <td>143.000000</td>
      <td>2.000000</td>
      <td>65.087816</td>
      <td>44.817314</td>
      <td>9.000000</td>
      <td>12.000000</td>
      <td>17.000000</td>
      <td>2002.000000</td>
      <td>0.195781</td>
      <td>35.000000</td>
      <td>7287.000000</td>
      <td>4.000000</td>
      <td>1450.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9998.000000</td>
      <td>209.000000</td>
      <td>19.000000</td>
      <td>641.065193</td>
      <td>638.163193</td>
      <td>2014.000000</td>
      <td>33.000000</td>
      <td>40.000000</td>
      <td>2020.000000</td>
      <td>0.521867</td>
      <td>74.000000</td>
      <td>19083.000000</td>
      <td>141.000000</td>
      <td>4702.000000</td>
      <td>6.000000</td>
      <td>23.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8000 entries, 0 to 7999
    Data columns (total 20 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             8000 non-null   int64  
     1   DistrictId     8000 non-null   int64  
     2   Rooms          8000 non-null   float64
     3   Square         8000 non-null   float64
     4   LifeSquare     6296 non-null   float64
     5   KitchenSquare  8000 non-null   float64
     6   Floor          8000 non-null   int64  
     7   HouseFloor     8000 non-null   float64
     8   HouseYear      8000 non-null   int64  
     9   Ecology_1      8000 non-null   float64
     10  Ecology_2      8000 non-null   object 
     11  Ecology_3      8000 non-null   object 
     12  Social_1       8000 non-null   int64  
     13  Social_2       8000 non-null   int64  
     14  Social_3       8000 non-null   int64  
     15  Healthcare_1   4155 non-null   float64
     16  Helthcare_2    8000 non-null   int64  
     17  Shops_1        8000 non-null   int64  
     18  Shops_2        8000 non-null   object 
     19  Price          8000 non-null   float64
    dtypes: float64(8), int64(9), object(3)
    memory usage: 1.2+ MB



```python
test_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2000 entries, 0 to 1999
    Data columns (total 19 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             2000 non-null   int64  
     1   DistrictId     2000 non-null   int64  
     2   Rooms          2000 non-null   float64
     3   Square         2000 non-null   float64
     4   LifeSquare     1591 non-null   float64
     5   KitchenSquare  2000 non-null   float64
     6   Floor          2000 non-null   int64  
     7   HouseFloor     2000 non-null   float64
     8   HouseYear      2000 non-null   int64  
     9   Ecology_1      2000 non-null   float64
     10  Ecology_2      2000 non-null   object 
     11  Ecology_3      2000 non-null   object 
     12  Social_1       2000 non-null   int64  
     13  Social_2       2000 non-null   int64  
     14  Social_3       2000 non-null   int64  
     15  Healthcare_1   1047 non-null   float64
     16  Helthcare_2    2000 non-null   int64  
     17  Shops_1        2000 non-null   int64  
     18  Shops_2        2000 non-null   object 
    dtypes: float64(7), int64(9), object(3)
    memory usage: 297.0+ KB



```python
plt.scatter(train_df.Square, train_df.Price)
plt.title('Square vs Price')
plt.xlabel('Square')
plt.ylabel('Price')
plt.show()
sns.despine
```


![png](output_11_0.png)





    <function seaborn.utils.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)>




```python
num_features = ['Rooms', 'Square', 'Floor', 'HouseFloor', 'HouseYear', 'KitchenSquare', 'Ecology_1', 'Shops_1', 'Helthcare_2']
```


```python
print(num_features)
```

    ['Rooms', 'Square', 'Floor', 'HouseFloor', 'HouseYear', 'KitchenSquare', 'Ecology_1', 'Shops_1', 'Helthcare_2']



```python
X_train = train_df[num_features].values
y_train = train_df['Price'].values
X_test = test_df[num_features].values
test_idxs = test_df['Id'].values
```


```python
np.nan_to_num(X_train)
print(X_train)
```

    [[1.00000000e+00 3.98325239e+01 7.00000000e+00 ... 1.18537385e-01
      0.00000000e+00 1.00000000e+00]
     [3.00000000e+00 7.83422151e+01 2.00000000e+00 ... 2.56091570e-02
      1.00000000e+00 3.00000000e+00]
     [1.00000000e+00 4.04099069e+01 1.00000000e+01 ... 7.12231700e-03
      1.00000000e+00 0.00000000e+00]
     ...
     [2.00000000e+00 6.43076843e+01 1.30000000e+01 ... 7.21575810e-02
      0.00000000e+00 0.00000000e+00]
     [1.00000000e+00 2.96480568e+01 3.00000000e+00 ... 4.60556389e-01
      5.00000000e+00 1.00000000e+00]
     [1.00000000e+00 3.23302924e+01 3.00000000e+00 ... 1.94489265e-01
      5.00000000e+00 3.00000000e+00]]



```python
np.nan_to_num(y_train)
print(y_train)
```

    [177734.55340714 282078.72085004 168106.00763001 ... 199715.14880702
     165953.91258031 171842.41185487]



```python
np.nan_to_num(X_test)
print(X_test)
```

    [[3.00000000e+00 1.15027311e+02 4.00000000e+00 ... 7.54236800e-02
      0.00000000e+00 0.00000000e+00]
     [1.00000000e+00 4.68878918e+01 1.20000000e+01 ... 7.12231700e-03
      1.00000000e+00 0.00000000e+00]
     [2.00000000e+00 5.39751436e+01 2.00000000e+00 ... 1.27375905e-01
      9.00000000e+00 3.00000000e+00]
     ...
     [1.00000000e+00 3.47239837e+01 6.00000000e+00 ... 5.21867054e-01
      0.00000000e+00 0.00000000e+00]
     [2.00000000e+00 5.09027238e+01 4.00000000e+00 ... 1.27811589e-01
      3.00000000e+00 1.00000000e+00]
     [1.00000000e+00 3.58154765e+01 9.00000000e+00 ... 1.27375905e-01
      9.00000000e+00 3.00000000e+00]]



```python
np.nan_to_num(test_idxs)
print(test_idxs)
```

    [1109 5304 7077 ... 6128 9764 2569]


Модель LinearRegression


```python
model = LinearRegression()
```


```python
model.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
y_pred = model.predict(X_test)
```


```python
print(y_pred)
```

    [301470.37985318 159390.70104397 232191.2519314  ... 116229.05303853
     195244.53493156 178922.55168498]



```python
submission_df = pd.DataFrame(zip(test_idxs, y_pred), columns = ['Id', 'Price'])
submission_df.to_csv('submission.csv', index=False)
```


```python
print(submission_df)
```

            Id          Price
    0     1109  301470.379853
    1     5304  159390.701044
    2     7077  232191.251931
    3     2199  141301.411635
    4     5205  184900.105779
    ...    ...            ...
    1995  7153  226607.989299
    1996  7729  178851.008312
    1997  6128  116229.053039
    1998  9764  195244.534932
    1999  2569  178922.551685
    
    [2000 rows x 2 columns]



```python
submission_df = pd.read_csv('submission.csv')
submission_df.head(15)
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
      <th>Id</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1109</td>
      <td>301470.379853</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5304</td>
      <td>159390.701044</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7077</td>
      <td>232191.251931</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2199</td>
      <td>141301.411635</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5205</td>
      <td>184900.105779</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3243</td>
      <td>191013.183022</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9200</td>
      <td>220489.463469</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9729</td>
      <td>371221.585996</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9271</td>
      <td>154773.685410</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1391</td>
      <td>158460.084354</td>
    </tr>
    <tr>
      <th>10</th>
      <td>8475</td>
      <td>195830.699048</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3384</td>
      <td>211088.191350</td>
    </tr>
    <tr>
      <th>12</th>
      <td>4996</td>
      <td>150696.338638</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2387</td>
      <td>297883.362796</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3616</td>
      <td>140820.187298</td>
    </tr>
  </tbody>
</table>
</div>



Модель Gradient Boosting for regression


```python
from sklearn import ensemble
```


```python
clf = ensemble.GradientBoostingRegressor(n_estimators=1474, max_depth=3, min_samples_split=2, learning_rate=0.1, loss='ls')
```


```python
clf.fit(X_train,y_train)
```




    GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                              init=None, learning_rate=0.1, loss='ls', max_depth=3,
                              max_features=None, max_leaf_nodes=None,
                              min_impurity_decrease=0.0, min_impurity_split=None,
                              min_samples_leaf=1, min_samples_split=2,
                              min_weight_fraction_leaf=0.0, n_estimators=1474,
                              n_iter_no_change=None, presort='deprecated',
                              random_state=None, subsample=1.0, tol=0.0001,
                              validation_fraction=0.1, verbose=0, warm_start=False)




```python
y_pred = clf.predict(X_test)
```


```python
print(y_pred)
```

    [220742.86496595 168101.21177691 264541.66100843 ... 147747.64388571
     193876.9798824  162349.10747063]



```python
t_sc = np.zeros((clf.n_estimators),dtype=np.float64)
```


```python
print(t_sc)
```

    [0. 0. 0. ... 0. 0. 0.]



```python
for i,y_pred in enumerate(clf.staged_predict(X_test)):
    t_sc[i]=clf.loss_(test_idxs,y_pred)
```


```python
testsc = np.arange((clf.n_estimators))+1
```


```python
print(testsc)
```

    [   1    2    3 ... 1472 1473 1474]



```python
print(t_sc)
```

    [4.39275314e+10 4.40030195e+10 4.41695678e+10 ... 4.96159813e+10
     4.96150448e+10 4.96217184e+10]



```python
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(testsc,clf.train_score_,'b-',label= 'Set dev train')
plt.plot(testsc,t_sc,'r-',label = 'set dev test')
```




    [<matplotlib.lines.Line2D at 0x7fcc4fe48590>]




![png](output_39_1.png)



```python
submission_df1 = pd.DataFrame(zip(test_idxs, y_pred), columns = ['Id', 'Price'])
submission_df1.to_csv('submission1.csv', index=False)
```


```python
submission_df1 = pd.read_csv('submission1.csv')
submission_df1.head(20)
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
      <th>Id</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1109</td>
      <td>220742.864966</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5304</td>
      <td>168101.211777</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7077</td>
      <td>264541.661008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2199</td>
      <td>158009.739075</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5205</td>
      <td>216369.628519</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3243</td>
      <td>268154.582956</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9200</td>
      <td>312435.871545</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9729</td>
      <td>421209.616615</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9271</td>
      <td>197762.906859</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1391</td>
      <td>183489.936680</td>
    </tr>
    <tr>
      <th>10</th>
      <td>8475</td>
      <td>236714.980021</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3384</td>
      <td>177389.323204</td>
    </tr>
    <tr>
      <th>12</th>
      <td>4996</td>
      <td>167805.439875</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2387</td>
      <td>275418.497782</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3616</td>
      <td>102295.086588</td>
    </tr>
    <tr>
      <th>15</th>
      <td>9884</td>
      <td>125926.476969</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1285</td>
      <td>194244.193296</td>
    </tr>
    <tr>
      <th>17</th>
      <td>6097</td>
      <td>357559.764552</td>
    </tr>
    <tr>
      <th>18</th>
      <td>3953</td>
      <td>188731.309424</td>
    </tr>
    <tr>
      <th>19</th>
      <td>4596</td>
      <td>188911.737993</td>
    </tr>
  </tbody>
</table>
</div>



Удаляем выбросы


```python
#Вычисляем строки со значениями-выбросами 
first_quartile = train_df.quantile(q=0.25)
third_quartile = train_df.quantile(q=0.75)
IQR = third_quartile - first_quartile
outliers = train_df[(train_df > (third_quartile + 1.5 * IQR)) | (train_df < (first_quartile - 1.5 * IQR))].count(axis=1)
outliers.sort_values(axis=0, ascending=False, inplace=True)

#Удаляем из датафрейма 400 строк, подходящих под критерии выбросов
outliers = outliers.head(400)
train_df.drop(outliers.index, inplace=True)
```


```python
print(first_quartile)
```

    Id                 2485.750000
    DistrictId           63.000000
    Rooms                 1.000000
    Square               41.800063
    LifeSquare           22.765329
    KitchenSquare         1.000000
    Floor                 4.000000
    HouseFloor            9.000000
    HouseYear          1974.000000
    Ecology_1             0.017647
    Social_1              6.000000
    Social_2           1564.000000
    Social_3              0.000000
    Healthcare_1        325.000000
    Helthcare_2           0.000000
    Shops_1               1.000000
    Price            153994.680334
    Name: 0.25, dtype: float64



```python
print(third_quartile)
```

    Id                 7470.500000
    DistrictId          145.000000
    Rooms                 2.000000
    Square               66.036608
    LifeSquare           45.204687
    KitchenSquare         9.000000
    Floor                12.000000
    HouseFloor           17.000000
    HouseYear          2001.000000
    Ecology_1             0.195781
    Social_1             36.000000
    Social_2           7227.000000
    Social_3              5.000000
    Healthcare_1       1548.000000
    Helthcare_2           2.000000
    Shops_1               5.000000
    Price            249970.954618
    Name: 0.75, dtype: float64



```python
print(IQR)
```

    Id                4984.750000
    DistrictId          82.000000
    Rooms                1.000000
    Square              24.236545
    LifeSquare          22.439358
    KitchenSquare        8.000000
    Floor                8.000000
    HouseFloor           8.000000
    HouseYear           27.000000
    Ecology_1            0.178134
    Social_1            30.000000
    Social_2          5663.000000
    Social_3             5.000000
    Healthcare_1      1223.000000
    Helthcare_2          2.000000
    Shops_1              4.000000
    Price            95976.274284
    dtype: float64



```python
print(outliers)
```

    1549    7
    671     6
    1941    6
    3581    5
    1186    5
           ..
    3073    2
    3071    2
    4078    2
    4085    2
    4099    2
    Length: 400, dtype: int64


Превращаем категорийные признаки в числовые


```python
#Вычисляем столбцы с категорийными признаками, затем заменяем их на числа
categorical_columns = train_df.columns[train_df.dtypes == 'object']
labelencoder = LabelEncoder()
for column in categorical_columns:
    train_df[column] = labelencoder.fit_transform(train_df[column])
    print(dict(enumerate(labelencoder.classes_)))

#Выводим сводную информацию о датафрейме и его столбцах (признаках), чтобы убедиться, что теперь они все содержат цифровые значения
train_df.info()
```

    {0: 'A', 1: 'B'}
    {0: 'A', 1: 'B'}
    {0: 'A', 1: 'B'}
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7600 entries, 0 to 7999
    Data columns (total 20 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             7600 non-null   int64  
     1   DistrictId     7600 non-null   int64  
     2   Rooms          7600 non-null   float64
     3   Square         7600 non-null   float64
     4   LifeSquare     5961 non-null   float64
     5   KitchenSquare  7600 non-null   float64
     6   Floor          7600 non-null   int64  
     7   HouseFloor     7600 non-null   float64
     8   HouseYear      7600 non-null   int64  
     9   Ecology_1      7600 non-null   float64
     10  Ecology_2      7600 non-null   int64  
     11  Ecology_3      7600 non-null   int64  
     12  Social_1       7600 non-null   int64  
     13  Social_2       7600 non-null   int64  
     14  Social_3       7600 non-null   int64  
     15  Healthcare_1   3893 non-null   float64
     16  Helthcare_2    7600 non-null   int64  
     17  Shops_1        7600 non-null   int64  
     18  Shops_2        7600 non-null   int64  
     19  Price          7600 non-null   float64
    dtypes: float64(8), int64(12)
    memory usage: 1.2 MB



```python
print(categorical_columns)
```

    Index(['Ecology_2', 'Ecology_3', 'Shops_2'], dtype='object')


Создаем целевую переменную, делим датасет на выборки


```python
#Назначаем целевой переменной цену всей квартиры
y = train_df['Price']

#Создаем список признаков, на основании которых будем строить модели
features = [
            'Rooms', 
            'Square', 
            'Floor', 
            'HouseFloor',
            'HouseYear',
            'KitchenSquare',
            'Ecology_1',
            'Social_1',
            'Social_2',
            'Social_3',
            'Helthcare_2',
            'Shops_1'
           ]

#Создаем датафрейм, состоящий из признаков, выбранных ранее
X = train_df[features]
np.nan_to_num(X)

#Проводим случайное разбиение данных на выборки для обучения (train) и валидации (val), по умолчанию в пропорции 0.75/0.25
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
```


```python
print(y)
```

    0       177734.553407
    1       282078.720850
    2       168106.007630
    3       343995.102962
    4       161044.944138
                ...      
    7995    136744.340827
    7996    119367.455796
    7997    199715.148807
    7998    165953.912580
    7999    171842.411855
    Name: Price, Length: 7600, dtype: float64


Модель Random forest


```python
#Создаем регрессионную модель случайного леса 
rf_model = RandomForestRegressor(n_estimators=2000, 
                                 n_jobs=-1,  
                                 bootstrap=False,
                                 criterion='mse',
                                 max_features=3,
                                 random_state=1,
                                 max_depth=55,
                                 min_samples_split=5
                                 )

#Проводим подгонку модели на обучающей выборке 
rf_model.fit(train_X, train_y)

#Вычисляем предсказанные значения цен на основе валидационной выборки
rf_prediction = rf_model.predict(val_X).round(0)

#Вычисляем и печатаем величины ошибок при сравнении известных цен квартир из валидационной выборки с предсказанными моделью
print_metrics(rf_prediction, val_y)
```

    
    R²: 0.69
    
    Средняя абсолютная ошибка: 15.1 %
    Медианная абсолютная ошибка: 7.72 %


Модель XGBoost


```python
#Создаем регрессионную модель XGBoost
xgb_model = xgb.XGBRegressor(objective ='reg:gamma', 
                             learning_rate = 0.1,
                             max_depth = 55, 
                             n_estimators = 2000,
                             nthread = -1,
                             eval_metric = 'gamma-nloglik', 
                             )

#Проводим подгонку модели на обучающей выборке 
xgb_model.fit(train_X, train_y)

#Вычисляем предсказанные значения цен на основе валидационной выборки
xgb_prediction = xgb_model.predict(val_X).round(0)

#Вычисляем и печатаем величины ошибок при сравнении известных цен квартир из валидационной выборки с предсказанными моделью
print_metrics(xgb_prediction, val_y)
```

Усреднение предсказаний моделей


```python
prediction = rf_prediction * 0.5 + xgb_prediction * 0.5
```


```python
print_metrics(prediction, val_y)
```

Изучаем важность признаков в модели Random forest


```python
#Рассчитываем важность признаков в модели Random forest
importances = rf_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

#Печатаем рейтинг признаков
print("Рейтинг важности признаков:")
for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

#Строим столбчатую диаграмму важности признаков
plt.figure()
plt.title("Важность признаков")
plt.bar(range(X.shape[1]), importances[indices], color="g", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
```

    Рейтинг важности признаков:
    1. Square (0.294883)
    2. Rooms (0.148057)
    3. Social_1 (0.089824)
    4. Social_2 (0.089762)
    5. Social_3 (0.076719)
    6. KitchenSquare (0.057378)
    7. HouseYear (0.053876)
    8. Ecology_1 (0.048762)
    9. HouseFloor (0.039993)
    10. Floor (0.037303)
    11. Shops_1 (0.034926)
    12. Helthcare_2 (0.028519)



![png](output_62_1.png)


Оцениваем квартиру


```python
flat = pd.DataFrame({
                     'Rooms':[5], 
                     'Square':[60],
                     'Floor':[5],
                     'HouseFloor':[5],
                     'HouseYear':[2014],
                     'KitchenSquare':[4],
                    'Ecology_1':[0.5],
                    'Social_1':[10],
                    'Social_2':[400],
                    'Social_3':[1],
                    'Helthcare_2':[5],
                    'Shops_1':[23]
                     })

#Вычисляем предсказанное значение стоимости по модели
rf_prediction_flat = rf_model.predict(flat).round(0)
xgb_prediction_flat = xgb_model.predict(flat).round(0)

#Усредняем полученные значения и умножаем на общую площадь квартиры
price = (rf_prediction_flat * 0.5 + xgb_prediction_flat * 0.5)*flat['Square'][0]

#Печатаем предсказанное значение цены квартиры
print(f'Предсказанная моделью цена квартиры: {int(price[0].round(-3))} рублей')
```

    Предсказанная моделью цена квартиры: 6419000 рублей



```python

```
