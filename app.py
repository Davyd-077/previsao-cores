import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(page_title="Previsor de Cores", layout="centered")

st.title("IA para Previsão de Cores")
st.write("Este app usa inteligência artificial para prever a próxima cor com base em um padrão de cores anterior.")

# Entrada do usuário
entrada_usuario = st.text_input("Insira a sequência de cores separadas por vírgula (ex: preto,branco,vermelho):")

# Função para treinar o modelo
def treinar_modelo(sequencia):
    le = LabelEncoder()
    sequencia_numerica = le.fit_transform(sequencia)

    dados = []
    for i in range(len(sequencia_numerica) - 3):
        entrada = sequencia_numerica[i:i+3]
        saida = sequencia_numerica[i+3]
        dados.append(entrada.tolist() + [saida])

    df = pd.DataFrame(dados, columns=['cor_1', 'cor_2', 'cor_3', 'proxima_cor'])
    X = df[['cor_1', 'cor_2', 'cor_3']]
    y = df['proxima_cor']

    modelo = RandomForestClassifier()
    modelo.fit(X, y)
    return modelo, le

# Processar e prever
if st.button("Prever próxima cor"):
    if entrada_usuario:
        cores = [c.strip().lower() for c in entrada_usuario.split(",") if c.strip()]
        if len(cores) < 4:
            st.warning("Insira pelo menos 4 cores para que a IA aprenda o padrão.")
        else:
            try:
                modelo, le = treinar_modelo(cores)
                ultimas_3 = cores[-3:]
                entrada = le.transform(ultimas_3).reshape(1, -1)
                predicao = modelo.predict(entrada)
                proxima_cor = le.inverse_transform(predicao)[0]
                st.success(f"A próxima cor prevista é: **{proxima_cor.upper()}**")
            except Exception as e:
                st.error(f"Erro ao processar os dados: {e}")
    else:
        st.warning("Por favor, insira uma sequência de cores.")
