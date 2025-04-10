from PIL import Image
import gradio as gr
import cv2

escalas_de_cores = ["RGB", "BGR", "HSV", "Gray", "LAB", "HLS", "YCrCB"]

def visibilidade_translacao():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True),  gr.update(visible=True),  gr.update(visible=True)

def visibilidade_rotacao():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True),  gr.update(visible=True)

def visibilidade_escala():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True),  gr.update(visible=True),  gr.update(visible=True)

def visibilidade_espaco():
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True),  gr.update(visible=True),  gr.update(visible=True)


with gr.Blocks() as demo:
    # Apresentação do aplicativo web
    gr.Markdown("# Bem-vind@ ao seu editor de imagem web :)")
    gr.Markdown("***")
    gr.Markdown("O programa foi desenvolvido por Marcos Aurélio para a disciplina de Visão Computacional - ECT3709")
    
    
    imagem = gr.Image(type="pil", label="Faça o upload da imagem aqui!")  
    
    # Botoes de efeitos
    gr.Markdown("<br>")
    gr.Markdown("## Escolha uma operação para ser aplicada na imagem")
    gr.Markdown("***")
    
    # Aplica todas as operações de uma vez
    enviar_todos = gr.Button(value="Aplicar todas as operações")
    gr.Markdown("***")
    
    
    with gr.Row():    
        with gr.Column():
            gr.Markdown("### Operações Geométricas")
            
            # Translação----------------------------------------------------------------------------------------------------------------
            translacao = gr.Button(value="Translação")
            translacao_x_text = gr.Textbox(label="Deslocamento Horizontal",visible=False, interactive=True)
            translacao_y_text = gr.Textbox(label="Deslocamento Vertical",visible=False, interactive=True)
        
            enviar_translacao = gr.Button(value="Aplicar operação", visible=False)
            voltar_menu_transformacoes_geometricas = gr.Button("Voltar", visible=False)

            # Rotação-------------------------------------------------------------------------------------------------------------------                    
            rotacao = gr.Button(value="Rotação")
            rotacao_slider = gr.Slider(minimum=0, maximum=360, label="Valor do ângulo", visible=False, interactive=True)
            enviar_rotacao = gr.Button(value="Aplicar operação", visible=False)
            
            # Escala-------------------------------------------------------------------------------------------------------------------                    
            escala = gr.Button(value="Escala")
            
            
        with gr.Column():
            gr.Markdown("### Operações de Cor")
            
            # Conversão de Espaços------------------------------------------------------------------------------------------------------
            conversao_espacos = gr.Button(value="Conversão de Espaços")
            espaco_origem = gr.Dropdown(choices=escalas_de_cores, visible=False, interactive=True)
            espaco_destino = gr.Dropdown(choices=escalas_de_cores, visible=False, interactive=True)
            
            enviar_espaco = gr.Button(value="Aplicar operação", visible=False)
            voltar_menu_conversao_cores = gr.Button("Voltar", visible=False)

             # Ajuste de Contraste------------------------------------------------------------------------------------------------------    
            ajuste_contraste = gr.Button(value="Ajuste de Contraste")    
        
        with gr.Column():
            gr.Markdown("### Correção Gama e Clareamento")
            correcao_gama = gr.Slider(minimum=0.1, maximum=3.0, label="Valor de Gamma", interactive=True)
            enviar_gama = gr.Button(value="Aplicar operação")

        # Interatividade dos grandes blocos
        # Propriedades Geometricas------------------------------------------------------------------------------------------------------
        translacao.click(fn=visibilidade_translacao, outputs=[rotacao, escala, translacao_x_text, translacao_y_text, enviar_translacao, voltar_menu_transformacoes_geometricas])
        rotacao.click(fn=visibilidade_rotacao, outputs=[translacao, escala, rotacao_slider, enviar_rotacao,  voltar_menu_transformacoes_geometricas])
        
        # Operações de cor--------------------------------------------------------------------------------------------------------------
        conversao_espacos.click(fn=visibilidade_espaco, outputs=[ajuste_contraste, espaco_origem, espaco_destino, enviar_espaco, voltar_menu_conversao_cores])


demo.launch()

"""
    Código desenvolvido por Marcos Aurélio Tavares Filho para a disciplina de Visão Computacional - ECT3709
"""
