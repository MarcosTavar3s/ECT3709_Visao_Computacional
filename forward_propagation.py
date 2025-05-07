import numpy as np
import gradio as gr

# Função sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Função ReLU
def relu(z):
  arr = []
  for zi in z:
    arr.append(max(0,zi))

  arr = np.array(arr)
  return arr

# Função Tanh
def tanh(z):
  return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

# Função LeakyReLU
def leaky_relu(z):
  alfa = 0.01
  arr = []
  for zi in z:
    arr.append(max(alfa*zi,zi))

  arr = np.array(arr)
  return arr

def def_function(name_function:str):
  name_function = name_function.lower()

  if(name_function == "r"):
    name_function = lambda z: relu(z)
  elif(name_function == "s"):
    name_function = lambda z: sigmoid(z)
  elif(name_function == "t"):
    name_function = lambda z: tanh(z)
  else:
    name_function = lambda z: leaky_relu(z)

  return name_function

def test(inputs, weights1, bias1, weights2, bias2, y_true, act_function1, act_function2):
  # Conversao do tipo - gradio vem em np com type string
  inputs = inputs.astype(float)
  weights1 = weights1.astype(float)
  bias1 = bias1.astype(float)
  weights2 = weights2.astype(float)
  bias2 = bias2.astype(float)
  y_true = y_true.astype(float)

  # first layer (2 neuronios)
  print(inputs.shape)
  print(weights1.shape)
  h_linear = weights1 @ inputs + bias1
  h = sigmoid(h_linear)


  # second layer - output (1 neuronio)
  y_predict = sigmoid(weights2 @ h + bias2)
  print(h, y_predict)

def main():
  inputs = np.array([[1], [2]])
  
  weights1 = np.array([[2, -1], [0,  3]])  
  weights2 = np.array([[3,5]])

  bias1 = np.array([[1], [-2]])
  bias2 = 5
  y_true = 29

  function1 = input("First Layer Config: Which activation function do you want to use?\nReLU (r), Sigmoid (s), Tanh (t), LeakyReLU (lr)\n")
  function2 = input("Second Layer Config: Which activation function do you want to use?\nReLU (r), Sigmoid (s), Tanh (t), LeakyReLU (lr)\n")

  function1 = def_function(function1)
  function2 = def_function(function2)

  
  # first layer (2 neurons)
  h_linear = weights1 @ inputs + bias1
  h = function1(h_linear)


  # second layer - output (1 neuron)
  y_predict = function2(weights2 @ h + bias2)

  # converte y para a escala da função
  # y_true = function(y_true)

  # error
  error = (y_predict-y_true) ** 2

  print(f"Inputs: {inputs}\n")
  print(f"Weights1: {weights1}\n")
  print(f"Weights2: {weights2}\n")
  print(f"First layer: {h}\n")
  print(f"Output (second layer): {y_predict}\n")
  print(f"True value: {y_true}\n")
  print(f"Error: {error}\n")


with gr.Blocks() as demo:
    gr.Markdown("# Bem-vind@ a simulação de rede neural artificial com 2 camadas :)")
    gr.Markdown("***")
    gr.Markdown("O programa foi desenvolvido por Marcos Aurélio para a disciplina de Visão Computacional - ECT3709. O programa consiste em uma rede neural com 2 neurônios na 1° camada e 1 neurônio na segunda.")
    
    gr.Markdown("## Definições dos parâmetros")
    gr.Markdown("***")
    
    with gr.Row():      
      inputs = gr.Dataframe(
        value=[[0], [0]],
        row_count="fixed",
        headers=["Entradas"], 
        type="numpy",
        interactive=True
    )
      
    with gr.Row():
      # 1° Camada da Rede
      bias1 = gr.DataFrame(
        value=[[0],[0]],
        row_count="fixed",
        headers=["Bias (1° camada)"], 
        type="numpy",
        interactive=True)
      
      weights1 = gr.DataFrame(
        value=[[0, 0], [0, 0]],
        row_count="fixed",
        headers=["Weights_col1 (1° camada)","Weights_col2 (1° camada)"], 
        type="numpy",
        interactive=True)
    
    with gr.Row():
    # 2° Camada da Rede
      bias2 = gr.DataFrame(
        value=[[0]],
        row_count="fixed",
        headers=["Bias (2° camada)"],  
        type="numpy",
        interactive=True)
      
      weights2 = gr.DataFrame(
        value=[[[0],[0]]],
        row_count="fixed",
        headers=["Weights_col1 (2° camada)","Weights_col2 (2° camada)"],  
        type="numpy",
        interactive=True)
    
  
    with gr.Row():
      y_true = gr.Textbox(label="Resultado esperado")
      
    with gr.Row():
      y_predict = gr.Text(value="", visible=False)
      error = gr.Text(value="", visible=False)
    
    act_function1 = gr.Dropdown(label="Escolha a função para a 1° camada da convolução", value="", choices=["sigmoid", "relu", "leaky_relu", "tanh"], interactive=True)
    act_function2 = gr.Dropdown(label="Escolha a função para a 2° camada da convolução", value="", choices=["sigmoid", "relu", "leaky_relu", "tanh"], interactive=True)
    
    enviar = gr.Button("Prever o resultado")
    enviar.click(fn=test,inputs=[inputs, weights1, bias1, weights1, bias2, y_true, act_function1, act_function2])
    # outputs=[inputs, weights1, bias1, weights2, bias2, y_true, act_function1, act_function2, y_predict, error]
    
demo.launch()
