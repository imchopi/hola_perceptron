import streamlit as st
import numpy as np

# Mostrar la imagen y el título
st.image("./img/neurona.jpg")
st.subheader("¡Hola perceptron!")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def binary_step(x):
    return 1 if x >= 0 else 0

# Selector de número de neuronas
num_neuronas = st.slider("Elige el número de entradas/pesos que tendrá la neurona", min_value=1, max_value=10, value=1, key="num_neuronas")

# Encabezados para los pesos y entradas en horizontal
st.write("### Pesos")
weight_cols = st.columns(num_neuronas)
st.write("### Entradas")
input_cols = st.columns(num_neuronas)

# Sesgo único para todas las neuronas
st.write("### Introduce el valor del sesgo")
b = st.number_input("Valor del sesgo", min_value=0.0, value=1.0, key="bias_global")

# Selección de función de activación única
st.write("### Función de activación única")
activation = st.selectbox("Elige la función de activación", ["Sigmoide", "ReLU", "Tangente hiperbólica", "Binary_step"], key="activation_global")

# Configurar cada neurona
y_total = b
for i in range(num_neuronas):
    with weight_cols[i]:
        w = st.number_input(f"w{i}", min_value=0.0, max_value=5.0, value=1.0, key=f"w_{i}")

    with input_cols[i]:
        x = st.number_input(f"x{i}", min_value=0.0, value=1.0, key=f"x_{i}")

    # Calcular la salida acumulada de las neuronas
    y_total += w * x

# Calcular y mostrar la salida total al presionar el botón
if st.button("Calcular la salida"):
    if activation == "Sigmoide":
        y_total = sigmoid(y_total)
    elif activation == "ReLU":
        y_total = relu(y_total)
    elif activation == "Tangente hiperbólica":
        y_total = tanh(y_total)
    elif activation == "Binary_step":
        y_total = binary_step(y_total)

    st.subheader("Resultado final")
    st.write(f"La salida de la neurona es {y_total}")
