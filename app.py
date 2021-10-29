import pickle 

from flask import Flask, render_template, request
app = Flask(__name__)

with open('vetorizador.pkl', 'rb') as file_vectorizer:
    vectorizer = pickle.load(file_vectorizer)

with open('classificador.pkl', 'rb') as file_classificador:
    classificador = pickle.load(file_classificador)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/classificador', methods=['POST'])
def cassificador():
    frase = request.form['texto']
    vetor = vectorizer.transform([frase])
    classificacao = 'Positivo' if classificador.predict(vetor)[0] == 1 else 'Negativo'
    return render_template('resultado.html', frase=frase, classificacao=classificacao)

if __name__ == '__main__':
    app.run(debug=True)
