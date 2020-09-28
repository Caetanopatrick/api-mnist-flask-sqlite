from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from keras.models import load_model
import tensorflow as tf
from datetime import datetime
import sqlite3

app = Flask(__name__, template_folder='templates')



def init():
    global model,graph
    # carregando o modelo criado
    model = load_model('model/mnistCNN.h5')
    graph = tf.get_default_graph()
#carregando a template no home
@app.route('/')
def upload_file():
   return render_template('index.html')
#carregando a imagem, transformando em array e prevendo com o modelo
@app.route('/uploader', methods = ['POST'])
def upload_image_file():
   if request.method == 'POST':
        img = Image.open(request.files['file'].stream).convert("L")
        img = img.resize((28,28))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,28,28,1)
        with graph.as_default():
            y_pred = model.predict_classes(im2arr)
        #salvando resultado em data.db
        connection = sqlite3.connect('data.db')
        cursor = connection.cursor()

        query = "INSERT INTO predictions VALUES (NULL,?, ?)"
        time = datetime.now().strftime("%B %d, %Y %I:%M%p")
        cursor.execute(query, (time, str(y_pred[0])) )
        connection.commit()
        connection.close()

        return 'Número previsto: ' + str(y_pred[0])




if __name__ == '__main__':
    print(("* Carregando modelo e Flask iniciando o server..."
        "por favor espere até que o server seja completamente iniciado"))
    init()
    app.run(debug = True)
