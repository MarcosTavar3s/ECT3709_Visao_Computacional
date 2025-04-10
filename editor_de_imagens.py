from PIL import Image
import gradio as gr
import cv2

escalas_de_cores = ["RGB", "BGR", "HSV", "Gray", "LAB", "HLS", "YCrCB"]

def visibilidade_translacao():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True),  gr.update(visible=True)

def visibilidade_rotacao():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

def visibilidade_escala():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True),  gr.update(visible=True)

def visibilidade_espaco():
    return gr.update(visible=True), gr.update(visible=True),  gr.update(visible=True)

def funcao_translacao():
    pass

def funcao_rotacao(img, angulo):
    if img is not None:    
        altura, largura = img.shape[:2]
        centro = (largura//2, altura//2)
        escala = 1.0 
        matriz_rotacao = cv2.getRotationMatrix2D(centro, angulo, escala)
        imagem_rotacionada = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        imagem_rotacionada = cv2.warpAffine(imagem_rotacionada, matriz_rotacao, (largura, altura))
        
        return imagem_rotacionada
    
    gr.Warning("Não foi anexada nenhuma imagem!", duration=5)
    return None
    
    
with gr.Blocks() as demo:
    # Apresentação do aplicativo web
    gr.Markdown("# Bem-vind@ ao seu editor de imagem web :)")
    gr.Markdown("***")
    gr.Markdown("O programa foi desenvolvido por Marcos Aurélio para a disciplina de Visão Computacional - ECT3709")
    
    
    imagem = gr.Image(type="numpy", label="Faça o upload da imagem aqui!")  
    
    # Botoes de efeitos
    gr.Markdown("<br>")
    gr.Markdown("## Escolha uma operação para ser aplicada na imagem")
    gr.Markdown("***")
    
    # Aplica todas as operações de uma vez
    # enviar_todos = gr.Button(value="Aplicar todas as operações")
    # gr.Markdown("***")
    
    
    with gr.Row():    
        with gr.Column():
            gr.Markdown("### Operações Geométricas")
            
            # Translação----------------------------------------------------------------------------------------------------------------
            translacao = gr.Button(value="Translação")
            translacao_x_text = gr.Textbox(label="Deslocamento Horizontal",visible=False, interactive=True)
            translacao_y_text = gr.Textbox(label="Deslocamento Vertical",visible=False, interactive=True)
        
            enviar_translacao = gr.Button(value="Aplicar operação", visible=False)
            # voltar_menu_transformacoes_geometricas = gr.Button("Voltar", visible=False)

            # Rotação-------------------------------------------------------------------------------------------------------------------                    
            rotacao = gr.Button(value="Rotação")
            rotacao_slider = gr.Slider(minimum=0, maximum=360, label="Valor do ângulo", visible=False, interactive=True)
            enviar_rotacao = gr.Button(value="Aplicar operação", visible=False)
            
            # Escala-------------------------------------------------------------------------------------------------------------------                    
            escala = gr.Button(value="Escala")
            escala_x_novo = gr.Textbox(label="Nova Largura",visible=False, interactive=True)
            escala_y_novo = gr.Textbox(label="Nova Altura",visible=False, interactive=True)
            
            enviar_escala = gr.Button(value="Aplicar operação", visible=False)
            
            
        with gr.Column():
            gr.Markdown("### Operações de Cor")
            
            # Ajuste de Contraste------------------------------------------------------------------------------------------------------    
            ajuste_contraste_slider = gr.Slider(label="Ajuste de Contraste")
            
            # Conversão de Espaços------------------------------------------------------------------------------------------------------
            conversao_espacos = gr.Button(value="Conversão de Espaços")
            espaco_origem = gr.Dropdown(choices=escalas_de_cores, visible=False, interactive=True)
            espaco_destino = gr.Dropdown(choices=escalas_de_cores, visible=False, interactive=True)
            
            enviar_espaco = gr.Button(value="Aplicar operação", visible=False)
            # voltar_menu_conversao_cores = gr.Button("Voltar", visible=False)

            
        
        with gr.Column():
            gr.Markdown("### Correção Gama e Clareamento")
            correcao_gama = gr.Slider(minimum=0.1, maximum=3.0, label="Valor de Gamma", interactive=True)
            enviar_gama = gr.Button(value="Aplicar operação")


        # Imagem após a operação--------------------------------------------------------------------------------------------------------
        imagem_final = gr.Image(label="Resultado", interactive=False, show_download_button=True)

        # Interatividade dos grandes blocos
        # Propriedades Geometricas------------------------------------------------------------------------------------------------------
        translacao.click(fn=visibilidade_translacao, outputs=[rotacao, escala, translacao_x_text, translacao_y_text, enviar_translacao])
        rotacao.click(fn=visibilidade_rotacao, outputs=[translacao, escala, rotacao_slider, enviar_rotacao])
        escala.click(fn=visibilidade_escala, outputs=[translacao, rotacao, escala_x_novo, escala_y_novo, enviar_escala])
        
        # Operações de cor--------------------------------------------------------------------------------------------------------------
        conversao_espacos.click(fn=visibilidade_espaco, outputs=[espaco_origem, espaco_destino, enviar_espaco])
        
        # Aplicação das operaçõe
        enviar_rotacao.click(fn=funcao_rotacao, inputs=[imagem, rotacao_slider], outputs=[imagem_final], )

demo.launch()

"""
    Código desenvolvido por Marcos Aurélio Tavares Filho para a disciplina de Visão Computacional - ECT3709
"""
