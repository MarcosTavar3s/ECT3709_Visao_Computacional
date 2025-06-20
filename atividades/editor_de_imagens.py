from PIL import Image
import gradio as gr
import numpy as np
import cv2

escalas_de_cores = ["RGB", "BGR", "HSV", "Gray", "LAB", "HLS", "YCrCB"]

def envia(img, translacao_x, translacao_y, rotacao_slider, ajuste_contraste_slider, correcao_gama, espaco_destino, escala_x_novo, escala_y_novo):
    if img is not None:
        # Pré-operação: normalização da imagem    
        # img = (img - np.min(img))/(np.max(img) - np.min(img))
        
        if translacao_x or translacao_y:
            img = funcao_translacao(img, translacao_x, translacao_y)
            
        if rotacao_slider:
            img = funcao_rotacao(img, rotacao_slider)
            
        if ajuste_contraste_slider:
            img = funcao_contraste(img, ajuste_contraste_slider)
            
        if correcao_gama:
            img = funcao_gama(img, correcao_gama)
            
        if escala_x_novo and escala_y_novo:
            img = funcao_escala(img, escala_x_novo, escala_y_novo)
    
        if espaco_destino != "RGB":
            img = funcao_conversao_cores(img, espaco_destino)
            
        return img, gr.update(value=None), gr.update(value=None), gr.update(value=0), gr.update(value=1), gr.update(value=1), gr.update(value="RGB"), gr.update(value=None), gr.update(value=None)
    
    gr.Warning("A imagem não foi carregada corretamente! Tente novamente!")
    return None

def funcao_translacao(img, deslocamento_x, deslocamento_y):
    if int(deslocamento_x) > img.shape[1] or int(deslocamento_y) > img.shape[0]:
        gr.Warning("O deslocamento vai além da dimensão da imagem! Operação não executada", duration=5)
        return None, gr.update(value=None), gr.update(value=None)
    
    matriz_translacao = np.float32([[1, 0, deslocamento_x],  # → controla eixo X  
                            [0, 1, deslocamento_y]]) # → controla eixo Y  
    
    imagem_transladada = cv2.warpAffine(img, matriz_translacao, (img.shape[1], img.shape[0]))
    
    return imagem_transladada

def funcao_rotacao(img, angulo):
    altura, largura = img.shape[:2]
    centro = (largura//2, altura//2)
    escala = 1.0 
    matriz_rotacao = cv2.getRotationMatrix2D(centro, angulo, escala)
        
    imagem_rotacionada = cv2.warpAffine(img, matriz_rotacao, (largura, altura))
    
    return imagem_rotacionada
    
def funcao_gama(img, gamma):
    imagem_gamma = (255*np.power(img/255., gamma)).astype(np.uint8)
    return imagem_gamma
        
def funcao_contraste(img, ajuste_contraste):   
    img_contraste = (ajuste_contraste * img).astype(np.uint8)
    return img_contraste

def funcao_escala(img, x_novo, y_novo):
    try:
        x_novo = int(x_novo)
        y_novo = int(y_novo)
        img = cv2.resize(img, (x_novo, y_novo))
    except:
        gr.Warning("Atributo inválido para redimensionar a imagem, coloque números inteiros!")
    return img

def funcao_conversao_cores(img, espaco_destino):
    
    if espaco_destino == "BGR":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif espaco_destino == "HSV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif espaco_destino == "Gray":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif espaco_destino == "LAB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    elif espaco_destino == "HLS":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif espaco_destino == "YrCrCB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        
    return img

with gr.Blocks() as demo:
    # Apresentação do aplicativo web
    gr.Markdown("# Bem-vind@ ao seu editor de imagem web :)")
    gr.Markdown("***")
    gr.Markdown("O programa foi desenvolvido por Marcos Aurélio para a disciplina de Visão Computacional - ECT3709")
    
    with gr.Row():
        # Imagem carregada pelo usuário-------------------------------------------------------------------------------------------------
        imagem = gr.Image(type="numpy", label="Faça o upload da imagem aqui!")  
        # Imagem após a operação--------------------------------------------------------------------------------------------------------
        imagem_final = gr.Image(label="Resultado", interactive=False, show_download_button=True)
        
    # Botoes de efeitos
    gr.Markdown("<br>")
    gr.Markdown("## Parâmetros de edição")
    gr.Markdown("***")
        
    
    with gr.Row():    
        with gr.Column():            
            # Translação----------------------------------------------------------------------------------------------------------------
            gr.Markdown("### Translação")
            translacao_x = gr.Number(maximum=4096, label="Deslocamento Horizontal", interactive=True)
            translacao_y = gr.Number(maximum=4096, label="Deslocamento Vertical", interactive=True)
        
            # Rotação-------------------------------------------------------------------------------------------------------------------                    
            gr.Markdown("### Rotação")
            rotacao_slider = gr.Slider(minimum=0, maximum=360, label="Valor do ângulo", interactive=True)
            
                     
        with gr.Column():
            # Escala-------------------------------------------------------------------------------------------------------------------                    
            gr.Markdown("#### Escala")
            escala_x_novo = gr.Textbox(label="Nova Largura", interactive=True)
            escala_y_novo = gr.Textbox(label="Nova Altura", interactive=True)
            
            gr.Markdown("### Ajuste de Contraste")
            
            # Ajuste de Contraste------------------------------------------------------------------------------------------------------    
            ajuste_contraste_slider = gr.Slider(value=1, minimum=0, maximum=3,label="Ajuste de Contraste")
            
        with gr.Column():
            # Correção Gama e Clareamento------------------------------------------------------------------------------------------------------
            gr.Markdown("### Correção Gama e Clareamento")
            correcao_gama = gr.Slider(value=1, minimum=0.1, maximum=3.0, label="Valor de Gamma", interactive=True)
            
            # Conversão de Espaços------------------------------------------------------------------------------------------------------
            gr.Markdown("### Conversão de Espaços")
            espaco_destino = gr.Dropdown(choices=escalas_de_cores, interactive=True)
            
    # Aplica todas as operações de uma vez
    enviar_todos = gr.Button(value="Aplicar todas as operações")
    enviar_todos.click(fn=envia, inputs=[imagem, translacao_x, translacao_y, rotacao_slider, ajuste_contraste_slider, correcao_gama, espaco_destino, escala_x_novo, escala_y_novo],
                       outputs=[imagem_final, translacao_x, translacao_y, rotacao_slider, ajuste_contraste_slider, correcao_gama, espaco_destino, escala_x_novo, escala_y_novo])
    
demo.launch()

"""
    Código desenvolvido por Marcos Aurélio Tavares Filho para a disciplina de Visão Computacional - ECT3709
"""
