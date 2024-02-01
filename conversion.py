
# In questo modulo definiamo la funzione 'arff_to_csv()', la quale sarà necessaria nei file
# 'Spam_detector.ipynb' e 'spam_detection_app.py' per importare i dati su cui lavorare.


import pandas as pd
import urllib.request
import re



def arff_to_csv(url: str, 
                columns: list,
                name_csv: str):

  """Definiamo una funzione che prende in input l'url in cui è presente un file formato 'arff',
  il nome degli attributi, e il nome con cui vogliamo salvare il file.
  Inoltre, restituisce un dataFrame pandas dopo averlo salvato come file .csv"""

  # Scarichiamo il file ARFF
  response = urllib.request.urlopen(url)

  # Otteniamo dati che verranno decodificati nella variabile 'data' come una stringa
  data = response.read().decode('latin-1')


  # Troviamo l'indice in cui iniziano i dati reali a partire dalla sequenza '@data\n'
  data_start_index = re.search(r'@data\n', data, re.IGNORECASE).end()


  # Ottieniamo i dati, i quali si trovano dopo l'indice prima individuato
  data = data[data_start_index:]


  # Dividiamo le righe dei dati
  lines = [line.strip() for line in data.split('\n') if line.strip()]
  data_rows = [line.split(',') for line in lines[0:]]
  

  # Creiamo un DataFrame Pandas
  df = pd.DataFrame(data_rows, columns=columns)


  # Salviamo i dati in un file csv
  df.to_csv(name_csv)


  return df
