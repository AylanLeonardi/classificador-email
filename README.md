# Classificador Automático de Emails

## Sobre
Sistema que classifica emails como Produtivo (precisa de ação) ou Improdutivo (apenas cortesia) usando IA da Hugging Face, e sugere respostas automáticas.

## Como Usar
1. Acesse o link da aplicação
2. Cole o texto do email ou faça upload de um arquivo .txt
3. Clique em "Processar Email"
4. Veja a classificação e a resposta sugerida

## Exemplos

Email Produtivo:
"Preciso do status da solicitação #123"
Resultado: Produtivo + resposta de suporte

Email Improdutivo:
"Obrigado pelo atendimento!"
Resultado: Improdutivo + resposta de agradecimento

## Tecnologias
- Python + Flask
- DeepSeek API (IA)
- HTML/CSS/JavaScript
- Render (hospedagem)

## Como rodar localmente
git clone https://github.com/AylanLeonardi/classificador-email.git

cd classificador-email

pip install -r requirements.txt

python app.py

Acesse: http://localhost:5000

## Autor
Aylan Leonardi Lima da Silva
