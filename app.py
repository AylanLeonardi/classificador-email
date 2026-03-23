from flask import Flask, request, jsonify, render_template
import requests
import re
import os 
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Baixar recursos do NLTK (executa apenas na primeira vez)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('rslp')

# Inicializa a aplicação Flask
app = Flask(__name__)

# Configuração da API de inferência do Hugging Face
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

# Obtém o token de autenticação das variáveis de ambiente
HF_TOKEN = os.getenv('HF_TOKEN', '')
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def classificar_com_ia(texto):
    """
    Classifica o conteúdo do e-mail utilizando modelo de IA do Hugging Face.
    
    Args:
        texto (str): Conteúdo do e-mail a ser classificado
        
    Returns:
        str: "Produtivo" ou "Improdutivo" baseado na classificação da IA
    """
    try:
        # Aplicar pré-processamento
        texto_processado = preprocessar_texto_completo(texto)
        
        # Usar texto processado na API
        texto_curto = texto_processado[:1000]
        
        # Prepara o payload para o modelo de classificação zero-shot
        payload = {
            "inputs": texto_curto,
            "parameters": {
                "candidate_labels": ["produtivo", "improdutivo"]
            }
        }
        
        # Realiza a requisição para a API do Hugging Face
        response = requests.post(API_URL, headers=headers, json=payload)
        
        # Processa a resposta se a requisição for bem-sucedida
        if response.status_code == 200:
            resultado = response.json()
            
            # Extrai a categoria com maior pontuação de confiança
            categoria_ia = resultado['labels'][0]
            score = resultado['scores'][0]
            
            # Registra o resultado da classificação para debug
            print(f"🤖 IA classificou como: {categoria_ia} (confiança: {score:.2%})")
            
            # Retorna a categoria em português conforme o padrão esperado
            if categoria_ia == "produtivo":
                return "Produtivo"
            else:
                return "Improdutivo"
                
        else:
            # Em caso de erro na API, utiliza o método de fallback
            print(f"❌ Erro na API: {response.status_code}")
            return classificar_fallback(texto)
            
    except Exception as e:
        # Captura e registra exceções durante o processamento
        print(f"❌ Erro ao chamar IA: {e}")
        return classificar_fallback(texto)

def classificar_fallback(texto):
    """
    Método alternativo de classificação baseado em palavras-chave.
    Utilizado quando o serviço de IA não está disponível.
    
    Args:
        texto (str): Conteúdo do e-mail a ser classificado
        
    Returns:
        str: "Produtivo" ou "Improdutivo" baseado em análise léxica
    """
    texto_lower = texto.lower()
    
    # Lista de palavras-chave associadas a e-mails produtivos
    acao = ['status', 'problema', 'ajuda', 'urgente', 'prazo', 
            'solicitação', 'aprovado', 'próxima fase', 'case']
    
    # Verifica se alguma palavra-chave está presente no texto
    if any(palavra in texto_lower for palavra in acao):
        return "Produtivo"
    return "Improdutivo"

def preprocessar_texto_completo(texto):
    """
    Pré-processamento completo com NLP
    """
    # 1. Remover caracteres especiais
    texto = re.sub(r'[^a-zA-Záéíóúâêôãõç\s]', ' ', texto)
    
    # 2. Converter para minúsculas
    texto = texto.lower()
    
    # 3. Separar em palavras
    palavras = texto.split()
    
    # 4. Remover stop words
    stop_words = set(stopwords.words('portuguese'))
    palavras_sem_stop = [p for p in palavras if p not in stop_words]
    
    # 5. Aplicar stemming
    stemmer = RSLPStemmer()
    palavras_stemizadas = [stemmer.stem(p) for p in palavras_sem_stop]
    
    # 6. Juntar tudo
    return ' '.join(palavras_stemizadas)

# Modelo de geração de texto
gen_url = "https://api-inference.huggingface.co/models/pierreguillou/gpt2-small-portuguese"

def gerar_resposta_com_ia(texto, categoria):
    """
    Gera resposta usando IA (ou fallback para template)
    """
    try:
        # Preparar o prompt baseado na categoria
        if categoria == "Produtivo":
            prompt = f"Responda profissionalmente este email de forma educada e útil: {texto[:200]}"
        else:
            prompt = f"Responda cordialmente este email de agradecimento: {texto[:150]}"
        
        # Chamar a API (usando o mesmo headers com seu token)
        response = requests.post(gen_url, headers=headers, json={
            "inputs": prompt,
            "parameters": {
                "max_length": 150,
                "temperature": 0.7,
                "do_sample": True
            }
        })
        
        # Verificar se funcionou
        if response.status_code == 200:
            resultado = response.json()
            resposta_ia = resultado[0]['generated_text']
            # Limpar a resposta (remover o prompt)
            resposta_ia = resposta_ia.replace(prompt, "").strip()
            
            if len(resposta_ia) > 20:
                return resposta_ia
                
    except Exception as e:
        print(f"Erro na geração com IA: {e}")
    
    # FALLBACK: templates fixos (caso a IA falhe)
    if categoria == "Produtivo":
        return f"""Olá! Recebemos sua mensagem: "{texto[:100]}..."

✅ Nossa equipe analisará e retornará em até 24 horas.

Atenciosamente,
Equipe de Suporte"""
    else:
        return """Olá! Agradecemos pelo seu contato! 😊

Estamos à disposição sempre que precisar.

Atenciosamente,
Equipe de Atendimento"""

@app.route('/')
def index():
    """
    Rota principal da aplicação.
    Renderiza a interface web para interação com o usuário.
    
    Returns:
        Template HTML renderizado
    """
    return render_template('index.html')

@app.route('/classificar', methods=['POST'])
def processar_email():
    """
    Endpoint responsável por processar e classificar e-mails.
    Aceita tanto texto direto quanto upload de arquivo .txt.
    
    Returns:
        JSON: Resultado da classificação e resposta gerada
    """
    try:
        conteudo = None
        
        # Verifica se houve upload de arquivo
        if 'arquivo' in request.files:
            arquivo = request.files['arquivo']
            if arquivo and arquivo.filename:
                # Valida extensão do arquivo
                if arquivo.filename.endswith('.txt'):
                    conteudo = arquivo.read().decode('utf-8')
                else:
                    return jsonify({'erro': 'Use arquivo .txt', 'sucesso': False}), 400
        
        # Caso não tenha arquivo, obtém texto do formulário
        if not conteudo:
            conteudo = request.form.get('texto', '')
        
        # Valida se o conteúdo não está vazio
        if not conteudo or not conteudo.strip():
            return jsonify({'erro': 'Email vazio', 'sucesso': False}), 400
        
        # Registra o e-mail recebido para debug
        print("\n" + "="*50)
        print(f"📧 EMAIL RECEBIDO:")
        print(f"{conteudo[:200]}...")
        print("="*50)
        
        # Realiza a classificação do e-mail
        categoria = classificar_com_ia(conteudo)
        
        # Gera resposta automática baseada na classificação
        resposta = gerar_resposta_com_ia(conteudo, categoria)
        
        # Registra o resultado do processamento
        print(f"📌 RESULTADO: {categoria}")
        print("="*50 + "\n")
        
        # Retorna o resultado em formato JSON
        return jsonify({
            'sucesso': True,
            'categoria': categoria,
            'resposta': resposta
        })
        
    except Exception as e:
        # Trata exceções não previstas
        print(f"❌ Erro: {e}")
        return jsonify({'erro': str(e), 'sucesso': False}), 500

if __name__ == '__main__':
    # Inicialização do servidor
    print("\n🚀 Servidor com IA iniciando...")
    print("📍 Acesse: http://localhost:5000")
    print("🤖 Usando modelo: facebook/bart-large-mnli")
    # Executa a aplicação em modo debug para desenvolvimento
    app.run(debug=True, host='0.0.0.0', port=5000)