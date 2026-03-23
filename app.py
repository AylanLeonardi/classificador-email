from flask import Flask, request, jsonify, render_template
import requests
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import re
import os
import PyPDF2
from dotenv import load_dotenv
import io

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Inicializa a aplicação Flask
app = Flask(__name__)

# Verifica e baixa recursos necessários do NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('rslp')

# Configuração da API DeepSeek
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

def extrair_texto_pdf(arquivo):
    """
    Extrai conteúdo textual de um arquivo PDF.
    
    Args:
        arquivo: Objeto de arquivo PDF (FileStorage)
        
    Returns:
        str: Texto extraído do PDF ou string vazia em caso de erro
    """
    try:
        pdf_reader = PyPDF2.PdfReader(arquivo)
        texto = ""
        # Itera sobre todas as páginas do PDF
        for page in pdf_reader.pages:
            texto += page.extract_text()
        return texto
    except Exception as e:
        print(f"Erro ao ler PDF: {e}")
        return ""

def preprocessar_texto(texto):
    """
    Aplica técnicas de Processamento de Linguagem Natural ao texto.
    
    Etapas:
    1. Remove caracteres especiais mantendo apenas letras e espaços
    2. Converte para minúsculas
    3. Remove stop words (palavras sem significado semântico)
    4. Aplica stemming para reduzir palavras à forma base
    
    Args:
        texto (str): Texto original a ser processado
        
    Returns:
        str: Texto processado
    """
    # Remove caracteres especiais, mantém apenas letras acentuadas e espaços
    texto = re.sub(r'[^a-zA-Záéíóúâêôãõç\s]', ' ', texto)
    # Converte para minúsculas
    texto = texto.lower()
    # Tokeniza o texto
    palavras = texto.split()
    
    try:
        # Remove stop words
        stop_words = set(stopwords.words('portuguese'))
        palavras_sem_stop = [p for p in palavras if p not in stop_words]
        # Aplica stemming
        stemmer = RSLPStemmer()
        palavras_stemizadas = [stemmer.stem(p) for p in palavras_sem_stop]
        return ' '.join(palavras_stemizadas)
    except:
        # Retorna texto original caso o processamento falhe
        return texto

def chamar_deepseek(prompt, max_tokens=200):
    """
    Realiza chamada para a API DeepSeek.
    
    Args:
        prompt (str): Instrução ou texto para processamento
        max_tokens (int): Número máximo de tokens na resposta
        
    Returns:
        str: Resposta gerada pela API ou None em caso de erro
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens
    }
    
    response = requests.post(DEEPSEEK_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        print(f"Erro na API: {response.status_code} - {response.text}")
        return None

def classificar_email(texto_original):
    """
    Classifica o e-mail como Produtivo ou Improdutivo.
    
    Critérios:
    - Produtivo: Requer ação, resposta, confirmação ou possui prazo
    - Improdutivo: Agradecimentos, felicitações ou mensagens sem necessidade de resposta
    
    Args:
        texto_original (str): Texto original do e-mail
        
    Returns:
        str: "Produtivo" ou "Improdutivo"
    """
    prompt = f"""
    Classifique o email abaixo como "Produtivo" ou "Improdutivo".
    
    - Produtivo: emails que exigem uma ação, resposta, confirmação, envio de informações, ou têm prazo.
    - Improdutivo: emails de agradecimento, felicitações, mensagens sem necessidade de resposta.
    
    Email:
    {texto_original[:800]}
    
    Responda APENAS com uma das palavras: Produtivo ou Improdutivo
    """
    
    resposta = chamar_deepseek(prompt, max_tokens=10)
    
    # Processa a resposta da API
    if resposta and "Produtivo" in resposta:
        return "Produtivo"
    elif resposta and "Improdutivo" in resposta:
        return "Improdutivo"
    # Valor padrão em caso de resposta inválida
    return "Produtivo"

def gerar_resposta(texto_original, categoria):
    """
    Gera resposta automática baseada na categoria do e-mail.
    
    Args:
        texto_original (str): Texto original do e-mail
        categoria (str): Categoria classificada ("Produtivo" ou "Improdutivo")
        
    Returns:
        str: Resposta gerada para o remetente
    """
    # Limita o tamanho do texto para processamento
    texto_curto = texto_original[:600]
    
    if categoria == "Produtivo":
        prompt = f"""
        Você é um assistente de suporte profissional. Analise o email abaixo e gere uma resposta APROPRIADA.
        
        IMPORTANTE: 
        - Identifique se o email é enviado PARA você (você é o destinatário) ou DE você (você é o remetente)
        - Se o email foi enviado PARA você, agradeça pelo contato e diga que está analisando
        - Se o email foi enviado DE você, você NÃO precisa responder
        - Agradeça pelo contato
        - Confirme que a mensagem foi recebida
        - Diga que a equipe está analisando
        - Mencione o prazo de retorno (24h)
        - Seja cordial, mas profissional
        - NÃO invente informações
        
        Email recebido:
        {texto_curto}
        
        Gere uma resposta profissional:
        """
    else:
        prompt = f"""
        Você é um assistente de suporte profissional. Gere uma resposta cordial.
        
        Email: {texto_curto}
        
        Resposta:
        """
    
    resposta = chamar_deepseek(prompt, max_tokens=150)
    
    # Valida e limpa a resposta gerada
    if resposta and len(resposta) > 20:
        resposta = resposta.replace('**', '')
        return resposta.strip()
    
    # Templates alternativos caso a API não retorne resposta válida
    if categoria == "Produtivo":
        return f"""Olá! Agradecemos pelo seu contato.

✅ Recebemos sua mensagem e nossa equipe já está analisando.

🔍 Retornaremos em até 24 horas.

Atenciosamente,
Equipe de Suporte"""
    else:
        return """Olá! 

😊 Agradecemos pela sua mensagem!

Estamos à disposição.

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
    Aceita texto direto ou upload de arquivos .txt e .pdf.
    
    Returns:
        JSON: Resultado da classificação e resposta gerada
    """
    try:
        conteudo = None
        
        # Processa upload de arquivo, se houver
        if 'arquivo' in request.files:
            arquivo = request.files['arquivo']
            if arquivo and arquivo.filename:
                # Verifica extensão do arquivo e extrai conteúdo
                if arquivo.filename.endswith('.txt'):
                    conteudo = arquivo.read().decode('utf-8')
                elif arquivo.filename.endswith('.pdf'):
                    conteudo = extrair_texto_pdf(arquivo)
                else:
                    return jsonify({'erro': 'Formato não suportado. Use .txt ou .pdf', 'sucesso': False}), 400
        
        # Caso não tenha arquivo, obtém texto do formulário
        if not conteudo:
            conteudo = request.form.get('texto', '')
        
        # Valida se o conteúdo não está vazio
        if not conteudo or not conteudo.strip():
            return jsonify({'erro': 'Email vazio', 'sucesso': False}), 400
        
        # Registra o e-mail recebido para debug
        print("\n" + "="*50)
        print(f"📧 EMAIL ORIGINAL: {conteudo[:100]}...")
        
        # Aplica pré-processamento NLP ao texto
        texto_processado = preprocessar_texto(conteudo)
        print(f"🔧 APÓS NLP: {texto_processado[:100]}...")
        print("="*50)
        
        # Realiza a classificação do e-mail
        categoria = classificar_email(conteudo)
        
        # Gera resposta automática baseada na classificação
        resposta = gerar_resposta(conteudo, categoria)
        
        # Registra o resultado do processamento
        print(f"📌 RESULTADO: {categoria}")
        print(f"💬 RESPOSTA: {resposta[:100]}...")
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
    print("\n🚀 Servidor com DeepSeek + NLP + PDF iniciando...")
    print("📍 Acesse: http://localhost:5000")
    # Executa a aplicação em modo debug para desenvolvimento
    app.run(debug=True, host='0.0.0.0', port=5000)