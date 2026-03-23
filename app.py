from flask import Flask, request, jsonify, render_template
import requests
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import re
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Baixar recursos do NLTK (primeira vez)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('rslp')

# DeepSeek API
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

def preprocessar_texto(texto):
    """
    Pré-processamento completo com NLP:
    - Remove caracteres especiais
    - Converte para minúsculas
    - Remove stop words
    - Aplica stemming
    """
    # 1. Remover caracteres especiais
    texto = re.sub(r'[^a-zA-Záéíóúâêôãõç\s]', ' ', texto)
    
    # 2. Converter para minúsculas
    texto = texto.lower()
    
    # 3. Separar em palavras
    palavras = texto.split()
    
    # 4. Remover stop words
    try:
        stop_words = set(stopwords.words('portuguese'))
        palavras_sem_stop = [p for p in palavras if p not in stop_words]
    except:
        palavras_sem_stop = palavras
    
    # 5. Aplicar stemming
    try:
        stemmer = RSLPStemmer()
        palavras_stemizadas = [stemmer.stem(p) for p in palavras_sem_stop]
    except:
        palavras_stemizadas = palavras_sem_stop
    
    # 6. Juntar tudo
    return ' '.join(palavras_stemizadas)

def chamar_deepseek(prompt, max_tokens=200):
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
        print(f"Erro: {response.status_code} - {response.text}")
        return None

def classificar_email(texto_original, texto_processado):
    """
    Classifica email usando DeepSeek
    Usa texto original para contexto (IA entende melhor)
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
    
    if resposta and "Produtivo" in resposta:
        return "Produtivo"
    elif resposta and "Improdutivo" in resposta:
        return "Improdutivo"
    return "Produtivo"

def gerar_resposta(texto_original, categoria):
    """Gera resposta personalizada baseada no contexto"""
    
    texto_curto = preprocessar_texto(texto_original)
    
    if categoria == "Produtivo":
        prompt = f"""
        Você é um assistente de suporte profissional. Analise o email abaixo e gere uma resposta APROPRIADA.
        
        IMPORTANTE: 
        - Identifique se o email é enviado PARA você (você é o destinatário) ou DE você (você é o remetente)
        - Se o email foi enviado PARA você, agradeça pelo contato e diga que está analisando
        - Se o email foi enviado DE você, você NÃO precisa responder (é um email que você enviou)
        - Neste caso, o email é enviado PARA você (Alan é o destinatário)
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
    
    if resposta and len(resposta) > 20:
        resposta = resposta.replace('**', '')
        return resposta.strip()
    
    # Fallback
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
    return render_template('index.html')

@app.route('/classificar', methods=['POST'])
def processar_email():
    try:
        conteudo = None
        
        if 'arquivo' in request.files:
            arquivo = request.files['arquivo']
            if arquivo and arquivo.filename and arquivo.filename.endswith('.txt'):
                conteudo = arquivo.read().decode('utf-8')
        
        if not conteudo:
            conteudo = request.form.get('texto', '')
        
        if not conteudo or not conteudo.strip():
            return jsonify({'erro': 'Email vazio', 'sucesso': False}), 400
        
        print("\n" + "="*50)
        print(f"📧 EMAIL ORIGINAL: {conteudo[:100]}...")
        
        # Aplicar pré-processamento
        texto_processado = preprocessar_texto(conteudo)
        print(f"🔧 APÓS NLP: {texto_processado[:100]}...")
        print("="*50)
        
        # Classificar
        categoria = classificar_email(conteudo, texto_processado)
        
        # Gerar resposta
        resposta = gerar_resposta(conteudo, categoria)
        
        print(f"📌 RESULTADO: {categoria}")
        print(f"💬 RESPOSTA: {resposta[:100]}...")
        print("="*50 + "\n")
        
        return jsonify({
            'sucesso': True,
            'categoria': categoria,
            'resposta': resposta
        })
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return jsonify({'erro': str(e), 'sucesso': False}), 500

if __name__ == '__main__':
    print("\n🚀 Servidor com DeepSeek + NLP iniciando...")
    print("📍 Acesse: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)