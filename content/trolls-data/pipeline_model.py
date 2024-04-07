
import os
import argparse
import logging
import pickle

import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def generate_tokenizer(train_csv, val_csv, test_csv, job_dir):

    train_df = pd.read_csv(train_csv, dtype=str)
    val_df = pd.read_csv(val_csv, dtype=str)
    test_df = pd.read_csv(test_csv, dtype=str)

    x_train = train_df.iloc[:, 0]
    x_val = val_df.iloc[:, 0]
    x_test = test_df.iloc[:, 0]

    y_train = train_df.iloc[:, 1]
    y_val = val_df.iloc[:, 1]
    y_test = test_df.iloc[:, 1]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)

    # Guardar el tokenizer como un archivo pickle
    tokenizer_path = os.path.join(job_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    train_sequences = tokenizer.texts_to_sequences(x_train)
    val_sequences = tokenizer.texts_to_sequences(x_val)
    test_sequences = tokenizer.texts_to_sequences(x_test)

    return tokenizer, train_sequences, val_sequences, test_sequences


# Actualiza tu función de entrenamiento para cargar los datos y generar el tokenizer

def train_model(epochs, job_dir, train_csv, val_csv, test_csv):
    tokenizer, train_sequences, val_sequences, test_sequences = generate_tokenizer(
      train_csv, val_csv, test_csv, job_dir)

    max_sequence_length = max([len(seq) for seq in train_sequences + val_sequences + test_sequences])
    train_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length, padding='post')
    val_sequences = pad_sequences(val_sequences, maxlen=max_sequence_length, padding='post')
    test_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length, padding='post')


    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_df = train_df.dropna()
    val_df = val_df.dropna()
    test_df = test_df.dropna()

    y_train = train_df.iloc[:, 1]
    y_val = val_df.iloc[:, 1]
    y_test = test_df.iloc[:, 1]


    model = MyModel()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

    model.fit(train_sequences, y_train, validation_data=(val_sequences, y_val), epochs=epochs)

    #model.save(os.path.join(job_dir, 'dp_model'))
    model.save(os.path(job_dir))

    loss, accuracy = model.evaluate(test_sequences, y_test)

    logging.info(f'Loss: {loss}, Accuracy: {accuracy}')




class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Definir las capas del modelo
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        # Definir la lógica de la propagación hacia adelante
        x = self.dense1(inputs)
        return self.dense2(x)



if __name__ == '__main__':
    # Configurar los argumentos de línea de comandos
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', dest='work_dir', required=True,
                        help='Directorio de trabajo para guardar el modelo entrenado')
    parser.add_argument('--epochs', dest='epochs', type=int, default=1,
                        help='Número de épocas de entrenamiento')
    parser.add_argument('--train-csv', dest='train_csv', required=True,
                        help='Ruta al archivo CSV de entrenamiento')
    parser.add_argument('--val-csv', dest='val_csv', required=True,
                        help='Ruta al archivo CSV de validación')
    parser.add_argument('--test-csv', dest='test_csv', required=True,
                        help='Ruta al archivo CSV de prueba')
    args = parser.parse_args()



    logging.basicConfig(level=logging.INFO)

    # Entreno el modelo
    train_model(args.epochs, args.work_dir, args.train_csv, args.val_csv, args.test_csv)

# Aquí introducimos la modificación para eliminar los nulos del resultado
input_csv_paths = [
    "./transformed_data/train/part-00000-of-00001.csv",
    "./transformed_data/val/part-00000-of-00001.csv",
    "./transformed_data/test/part-00000-of-00001.csv"
]

output_csv_paths = [
    "./transformed_data/train/part-00000-of-00001.csv",
    "./transformed_data/val/part-00000-of-00001.csv",
    "./transformed_data/test/part-00000-of-00001.csv"
]


for input_csv_path, output_csv_path in zip(input_csv_paths, output_csv_paths):
    df = pd.read_csv(input_csv_path)
    df = df.dropna()
    df.to_csv(output_csv_path, index=False)

print("Valores nulos eliminados y archivos CSV actualizados correctamente.")
