## From https://github.com/konstantinklemmer/pe-gnn
import io
import requests
from urllib import request 
from zipfile import ZipFile
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn.datasets
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import geopandas as gpd




def get_election_data(norm_x=True,norm_y=True):
  '''
  Download and process the Election dataset used in CorrelationGNN (https://arxiv.org/abs/2002.08274)

  Parameters:
  norm_x = logical; should features be normalized
  norm_y = logical; should outcome be normalized
  

  Return:
  data_train
  data_test
  '''

  Path("./election_data").mkdir(parents=True, exist_ok=True)
  zipurl = 'https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_counties_national.zip'
  with request.urlopen(zipurl) as zipresp:
      with ZipFile(io.BytesIO(zipresp.read())) as zfile:
          zfile.extractall('./election_data')

  geo = pd.read_csv("./election_data/2020_Gaz_counties_national.txt",sep='\t')
  geo = geo.rename(columns={"GEOID":"FIPS",'INTPTLONG                                                                                                               ':'INTPTLONG'})

  url = 'https://raw.githubusercontent.com/000Justin000/gnn-residual-correlation/master/datasets/election/education.csv'
  url_open = request.urlopen(url)
  edu = pd.read_csv(io.StringIO(url_open.read().decode('utf-8'))) 

  url = 'https://raw.githubusercontent.com/000Justin000/gnn-residual-correlation/master/datasets/election/election.csv'
  url_open = request.urlopen(url)
  ele = pd.read_csv(io.StringIO(url_open.read().decode('utf-8'))) 
  ele = ele.rename(columns={"fips_code":"FIPS"})

  url = 'https://raw.githubusercontent.com/000Justin000/gnn-residual-correlation/master/datasets/election/income.csv'
  url_open = request.urlopen(url)
  inc = pd.read_csv(io.StringIO(url_open.read().decode('utf-8'))) 

  url = 'https://raw.githubusercontent.com/000Justin000/gnn-residual-correlation/master/datasets/election/unemployment.csv'
  url_open = request.urlopen(url)
  une = pd.read_csv(io.StringIO(url_open.read().decode('utf-8'))) 

  url = 'https://raw.githubusercontent.com/000Justin000/gnn-residual-correlation/master/datasets/election/population.csv'
  url_open = request.urlopen(url)
  pop = pd.read_csv(io.StringIO(url_open.read().decode('utf-8'))) 

  dfs = [geo,edu,ele,inc,une,pop]
  data = reduce(lambda  left,right: pd.merge(left,right,on=['FIPS'],how='outer'), dfs)
  data = data.replace({',':''}, regex=True)

  out_data = np.array([data.INTPTLONG,data.INTPTLAT,data.dem_2016,data.gop_2016,data.MedianIncome2016,data.R_NET_MIG_2016,data.R_birth_2016,data.R_death_2016,data.BachelorRate2016,data.Unemployment_rate_2016]).T.astype(float)
  out_data = out_data[~np.isnan(out_data).any(axis=1)]
  out_data = out_data[(out_data[:,0] > -130) & (out_data[:,0] < -50) & (out_data[:,1] > 22) & (out_data[:,1] < 50)]

  coords = out_data[:,:2]

  y = out_data[:,3]
  x = out_data[:,[2,4,5,6,7,8,9]]
  df = pd.DataFrame(x,columns=["dem_2016","MedianIncome2016","R_NET_MIG_2016","R_birth_2016","R_death_2016","BachelorRate2016","Unemployment_rate_2016"])
  df['gop_2016'] = y
  df['x']= coords[:,0]
  df['y']= coords[:,1]

  data_train, data_test = train_test_split(df, test_size=0.3, shuffle=True, random_state=0)
  data_val, data_test = train_test_split(data_test, test_size=0.5, shuffle=True,  random_state=0)
  

  if norm_y==True:
    scaler = MinMaxScaler()
    data_train['gop_2016'] = scaler.fit_transform(data_train['gop_2016'].values.reshape(-1,1))
    data_val['gop_2016'] = scaler.transform(data_val['gop_2016'].values.reshape(-1,1))
    data_test['gop_2016'] = scaler.transform(data_test['gop_2016'].values.reshape(-1,1))
  if norm_x==True:
    data_train.iloc[:,:-3] = scaler.fit_transform(data_train.iloc[:,:-3])
    data_val.iloc[:,:-3] = scaler.transform(data_val.iloc[:,:-3])
    data_test.iloc[:,:-3] = scaler.transform(data_test.iloc[:,:-3])

  return data_train, data_val, data_test

def get_3d_road_data(norm_y=True, sample=1):
  '''
  Download and process the 3d road dataset

  Parameters:
  
  norm_y = logical; should outcome be normalized

  Return:
  data_train
  data_test
  '''
  # Both of the above sources contain the 3d road dataset
  url="https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt"
  #url="http://nrvis.com/data/mldata/3D_spatial_network.csv"
  s=requests.get(url).content
  df=pd.read_csv(io.StringIO(s.decode('utf-8')))
  df.columns = ["id","x","y","z"]
  df = df.drop_duplicates(subset=['x','y'])
  if sample <1.:
    df = df.sample(frac=sample, random_state=0)


  df.drop(columns=['id'],inplace=True)

  data_train, data_test = train_test_split(df, test_size=0.3, shuffle=True, random_state=0)
  data_val, data_test = train_test_split(data_test, test_size=0.5, shuffle=True, random_state=0)
  if norm_y==True:
    scaler = MinMaxScaler()
    data_train['z'] = scaler.fit_transform(data_train['z'].values.reshape(-1,1))
    data_val['z'] = scaler.transform(data_val['z'].values.reshape(-1,1))
    data_test['z'] = scaler.transform(data_test['z'].values.reshape(-1,1))


  return data_train, data_val, data_test

def get_air_temp_data(norm_y=True,norm_x=True):
  '''
  Download and process the Global Air Temperature dataset

  Parameters:
  norm_x = logical; should features be normalized
  norm_y = logical; should outcome be normalized
  
  Return:
  data_train
  data_test
  '''
  url = 'https://springernature.figshare.com/ndownloader/files/12609182'
  url_open = request.urlopen(url)
  df = pd.read_csv(io.StringIO(url_open.read().decode('utf-8')))
  df = df[['Lat','Lon','meanT','meanP']]
  df.rename(columns={"Lat":"y","Lon":"x"},inplace=True)
  



  data_train, data_test = train_test_split(df, test_size=0.3, shuffle=True, random_state=0)
  data_val, data_test = train_test_split(data_test, test_size=0.5, shuffle=True, random_state=0)

  if norm_y==True:
    scaler = MinMaxScaler()
    data_train['meanT'] = scaler.fit_transform(data_train['meanT'].values.reshape(-1,1))
    data_val['meanT'] = scaler.transform(data_val['meanT'].values.reshape(-1,1))
    data_test['meanT'] = scaler.transform(data_test['meanT'].values.reshape(-1,1))
  if norm_x==True:
    scaler = MinMaxScaler()
    data_train['meanP']= scaler.fit_transform(data_train['meanP'].values.reshape(-1,1))
    data_val['meanP'] = scaler.transform(data_val['meanP'].values.reshape(-1,1))
    data_test['meanP'] = scaler.transform(data_test['meanP'].values.reshape(-1,1))

  return data_train, data_val, data_test

