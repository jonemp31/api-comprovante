"""
API de Extração e Validação de Comprovantes PIX
================================================
Extrai dados estruturados de comprovantes PIX (imagem/PDF) usando OCR + regex.
Calcula trust score baseado em regras determinísticas.

Bancos suportados: Nubank, Banco do Brasil, PagBank, Mercado Pago, 
                   Cresol, Itaú, Banco Inter, Caixa, e genérico.

Uso:
    uvicorn app:app --host 0.0.0.0 --port 8000
    
    POST /extract  (multipart/form-data, campo "file")
    POST /extract/base64  (JSON: {"file": "base64...", "filename": "comprovante.jpg"})
"""

import re
import io
import os
import base64
import hashlib
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional
from pathlib import Path

import httpx
from PIL import Image, ImageOps
import fitz  # pymupdf
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pix-api")

app = FastAPI(
    title="PIX Receipt Extractor API",
    description="Extrai dados de comprovantes PIX e calcula trust score",
    version="1.0.0",
)

# ============================================================
# DADOS ESPERADOS DO RECEBEDOR (via variáveis de ambiente)
# ============================================================

EXPECTED_NOME_RECEBEDOR = os.getenv("EXPECTED_NOME_RECEBEDOR", "Maria Eduarda Cavaton")
EXPECTED_CPF_RECEBEDOR_PARTIAL = os.getenv("EXPECTED_CPF_RECEBEDOR_PARTIAL", "824.458")
EXPECTED_INSTITUICAO_RECEBEDOR = os.getenv("EXPECTED_INSTITUICAO_RECEBEDOR", "Banco Inter")
EXPECTED_CHAVE_PIX = os.getenv("EXPECTED_CHAVE_PIX", "16991500219")
MAX_HORAS_VALIDADE = int(os.getenv("MAX_HORAS_VALIDADE", "24"))


# Fuso horário de Brasília (UTC-3)
BRT = timezone(timedelta(hours=-3))

# Registro de IDs de transação processados (detecção de duplicatas)
_processed_transaction_ids: dict[str, float] = {}  # {id: timestamp}
_DUPLICATE_TTL_HOURS = 720  # 30 dias

logger.info("=== Configuração carregada ===")
logger.info(f"  EXPECTED_NOME_RECEBEDOR = {EXPECTED_NOME_RECEBEDOR}")
logger.info(f"  EXPECTED_CPF_RECEBEDOR_PARTIAL = {EXPECTED_CPF_RECEBEDOR_PARTIAL}")
logger.info(f"  EXPECTED_INSTITUICAO_RECEBEDOR = {EXPECTED_INSTITUICAO_RECEBEDOR}")
logger.info(f"  EXPECTED_CHAVE_PIX = {EXPECTED_CHAVE_PIX}")
logger.info(f"  MAX_HORAS_VALIDADE = {MAX_HORAS_VALIDADE}h")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODELS
# ============================================================

class PixData(BaseModel):
    banco_origem: Optional[str] = None
    nome_pagador: Optional[str] = None
    cpf_pagador: Optional[str] = None
    instituicao_pagador: Optional[str] = None
    nome_recebedor: Optional[str] = None
    cpf_recebedor: Optional[str] = None
    instituicao_recebedor: Optional[str] = None
    chave_pix: Optional[str] = None
    valor: Optional[float] = None
    valor_raw: Optional[str] = None
    data_hora: Optional[str] = None
    id_transacao: Optional[str] = None
    tipo: str = "pix"
    raw_text: Optional[str] = None


class TrustScore(BaseModel):
    score: float  # 0.0 - 1.0
    nivel: str  # "alto", "medio", "baixo", "suspeito"
    detalhes: list[str]
    penalidades: list[str]


class ExtractionResult(BaseModel):
    success: bool
    dados: Optional[PixData] = None
    trust: Optional[TrustScore] = None
    error: Optional[str] = None


class Base64Input(BaseModel):
    file: str
    filename: str = "comprovante.jpg"


class UrlInput(BaseModel):
    url: str
    filename: str = ""


# ============================================================
# OCR ENGINE — Tesseract
# ============================================================

import pytesseract


def preprocess_image(img: Image.Image) -> Image.Image:
    """Pré-processamento para melhorar OCR: resize 2x, grayscale, autocontrast."""
    w, h = img.size
    img = img.resize((w * 2, h * 2), Image.LANCZOS)
    img = img.convert('L')
    img = ImageOps.autocontrast(img)
    return img


def extract_text_from_image(image_bytes: bytes) -> str:
    """Extrai texto de uma imagem usando Tesseract OCR."""
    img = Image.open(io.BytesIO(image_bytes))
    img = preprocess_image(img)
    text = pytesseract.image_to_string(img, lang='por')
    return text


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extrai texto de PDF. Tenta texto nativo primeiro, depois OCR."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""

    for page in doc:
        # Tenta extrair texto nativo
        text = page.get_text()
        if text.strip():
            full_text += text
        else:
            # PDF é imagem - renderiza e faz OCR
            pix = page.get_pixmap(dpi=250)
            img_bytes = pix.tobytes("png")
            full_text += extract_text_from_image(img_bytes)

    doc.close()
    return full_text


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Roteador: escolhe extrator baseado no tipo de arquivo."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    else:
        return extract_text_from_image(file_bytes)


# ============================================================
# HELPERS DE REGEX
# ============================================================

def clean_text(text: str) -> str:
    """Normaliza texto OCR."""
    # Remove null bytes e caracteres de controle (comum em PDFs do BB)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    # Normaliza caracteres PUA de fontes customizadas (PicPay usa \ue092=: \ue088=- \ue09d=+)
    text = text.replace('\ue092', ':').replace('\ue088', '-').replace('\ue09d', '+')
    # Normaliza bullet/dot chars que PDFs usam para mascarar CPF
    text = text.replace('\u2022', '*')  # bullet → *
    text = text.replace('\u2023', '*')  # triangular bullet → *
    text = text.replace('\u25cf', '*')  # black circle → *
    text = text.replace('\u00b7', '*')  # middle dot → *
    # Corrige artefatos comuns do Tesseract
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    return text


def _nomes_correspondem(nome_extraido: str, nome_esperado: str) -> bool:
    """Compara nomes com tolerancia a erros de OCR/extracao de PDF.
    Aceita se pelo menos 2 palavras significativas (>=3 chars) do nome extraido
    aparecem no nome esperado.
    """
    if not nome_extraido or not nome_esperado:
        return False
    extraido_lower = nome_extraido.lower()
    esperado_lower = nome_esperado.lower()
    # Substring exata (caso ideal)
    if esperado_lower in extraido_lower or extraido_lower in esperado_lower:
        return True
    # Word-based: palavras significativas (>= 3 chars)
    palavras_extraidas = [w for w in re.split(r'\s+', extraido_lower) if len(w) >= 3]
    palavras_esperadas = [w for w in re.split(r'\s+', esperado_lower) if len(w) >= 3]
    if not palavras_extraidas or not palavras_esperadas:
        return False
    matches = sum(1 for p in palavras_extraidas if p in palavras_esperadas)
    return matches >= 2 or (matches >= 1 and len(palavras_extraidas) == 1)


def find_cpf(text: str, context_hint: str = "") -> Optional[str]:
    """Encontra CPF no texto, com ou sem máscara."""
    patterns = [
        r'(\d{3}\.?\d{3}\.?\d{3}-?\d{2})',  # CPF completo
        r'(\*{2,3}\.?\d{3}\.?\d{3}-?\*{2})',  # Mascarado ***. xxx.xxx-**
        r'(\*+\.?\d{3}\.?\d{3}[\-\.]\*+)',  # Variações de máscara
        r'([\*\.]+\d{3}\.\d{3}[\-\.][\*\.]+)',  # ···824.458-··
        r'(\w*\d{3}[\.\s]?\d{3}\w*)',  # Fallback mais relaxado
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            for m in matches:
                # Filtra matches que parecem CPF (tem pelo menos 6 dígitos ou asteriscos+dígitos)
                digits = re.findall(r'\d', m)
                masks = re.findall(r'\*', m)
                if len(digits) >= 6 or (len(digits) >= 3 and len(masks) >= 2):
                    return m.strip()
    return None


def find_valor(text: str) -> tuple[Optional[float], Optional[str]]:
    """Encontra valor monetário no texto."""
    patterns = [
        r'R\$\s*([\d\.]+,\d{2})',  # R$ 20,00
        r'R\$\s*(\d+)',  # R$ 50 (sem centavos)
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            raw = match.group(1)
            # Converte para float
            valor_str = raw.replace(".", "").replace(",", ".")
            try:
                return float(valor_str), f"R$ {raw}"
            except ValueError:
                continue
    return None, None


def _fix_ocr_digits(text: str) -> str:
    """Corrige letras comuns confundidas com dígitos pelo OCR em contextos numéricos."""
    # O/o → 0, l/I → 1 quando adjacente a dígitos
    result = text
    result = re.sub(r'(?<=\d)[Oo](?=\d)', '0', result)
    result = re.sub(r'[Oo](?=\d{1,2}\s+\w{3}\s+\d{4})', '0', result)  # O8 JAN 2026
    result = re.sub(r'(?<=\d)[lI](?=\d)', '1', result)
    return result


def find_data(text: str) -> Optional[str]:
    """Encontra data/hora no texto."""
    # Tenta com texto original e com correção de OCR
    for t in [text, _fix_ocr_digits(text)]:
        patterns = [
            r'(\d{2}/\d{2}/\d{4}\s+(?:às\s+)?\d{2}:\d{2}:\d{2})',
            r'(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2})',
            r'(\d{2}\s+\w+\s+\d{4}\s*[-–]\s*\d{2}:\d{2}:\d{2})',  # 07 DEZ 2025 - 07:02:21
            r'(\d{2}\s+\w{3}\s+\d{4}\s*[-–]\s*\d{2}:\d{2}:\d{2})',
            r'[Ss]ábado,?\s*(\d{1,2}\s+de\s+\w+\s+de\s+\d{4},?\s+às\s+\d{2}:\d{2}:\d{2})',
            r'(\d{2}/\w{3}/\d{4}\s*[-–]\s*\d{2}:\d{2}:\d{2})',  # 02/mar/2026 - 19:44:50
            r'(\d{2}/\w{3}/\d{4}\s*[-–]\s*\d{2}:\d{2})',  # 02/mar/2026 - 19:44
            r'(\d{1,2}/\w{3,10}/\d{4}\s+às\s+\d{1,2}h\d{2})',  # 17/março/2026 às 17h00
            r'(\d{2}/\d{2}/\d{4})',  # Só data
            r'(\d{1,2}\s+\w{3}\s+\d{4},\s*\d{1,2}:\d{2}(?::\d{2})?)',  # 18 mar 2026, 3:50:48
            r'(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})',  # 17 de março de 2026
        ]
        for pattern in patterns:
            match = re.search(pattern, t)
            if match:
                return match.group(1).strip()
    return None


def find_id_transacao(text: str) -> Optional[str]:
    """Encontra ID de transação PIX (padrão E + 32 chars do BACEN)."""
    # Padrão BACEN: E + ISPB(8) + data + ID
    patterns = [
        r'(E\d{8,14}\d{8,}[\w]+)',  # ID PIX padrão
        r'ID[:\s]+\n?(E[\w]{20,})',
        r'(?:ID\s+(?:de\s+)?transação\s*(?:PIX)?[:\s]*\n?\s*)(E[\w]{20,})',
        r'(?:Código\s+da\s+transação\s+Pix[:\s]*\n?\s*)(E[\w]{20,})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _normalize_chave_pix(chave: str) -> str:
    """Normaliza chave PIX removendo formatação: +55, (), -, espaços, *."""
    # Remove +55 prefix
    cleaned = re.sub(r'^\+?55\s*', '', chave.strip())
    # Remove formatting chars
    cleaned = re.sub(r'[\(\)\-\s\*]', '', cleaned)
    return cleaned


def find_chave_pix(text: str) -> Optional[str]:
    """Encontra chave PIX no texto."""
    patterns = [
        r'[Cc]h(?:ave)?\s*(?:Pix)?[:\s]+(\+?\d{10,13})',  # Telefone
        r'[Cc]h(?:ave)?\s*(?:Pix)?[:\s]+\((\d{2})\)\s*(\d{4,5}[-\s]?\d{4})',  # (16) 99150-0219
        r'[Cc]h(?:ave)?[:\s]+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',  # Email
        r'[Cc]h(?:ave)?[:\s]+(\d{3}\.?\d{3}\.?\d{3}-?\d{2})',  # CPF
        r'[Cc]h(?:ave)?[:\s]+(\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2})',  # CNPJ
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                return f"({groups[0]}) {groups[1]}"
            return groups[0].strip()
    return None


def find_nome(text: str, section: str, labels: list[str]) -> Optional[str]:
    """Encontra nome em uma seção do texto."""
    for label in labels:
        pattern = rf'{label}[:\s]*\n?\s*([A-ZÀ-Ú][A-ZÀ-Ú\sa-zà-ú\.]+)'
        match = re.search(pattern, section)
        if match:
            nome = match.group(1).strip()
            # Limpa artefatos
            nome = re.sub(r'\s+', ' ', nome)
            # Remove palavras que são labels seguintes
            stop_words = ['CPF', 'CNPJ', 'Instituição', 'Agência', 'Conta', 'Chave', 'Tipo']
            for sw in stop_words:
                idx = nome.find(sw)
                if idx > 0:
                    nome = nome[:idx].strip()
            return nome if len(nome) > 2 else None
    return None


# ============================================================
# CLASSIFICAÇÃO DE BANCO
# ============================================================

def classify_bank(text: str) -> str:
    """Identifica o banco emissor do comprovante."""
    text_lower = text.lower()
    # Normaliza OCR artifacts para melhorar detecção
    text_norm = re.sub(r'[\s]+', ' ', text_lower)

    if 'neon' in text_lower or 'neon pagamentos' in text_lower:
        return 'neon'
    if any(kw in text_lower for kw in ['comprovante bb', 'bco do brasil', 'banco do brasil', 'bco brasil']):
        return 'banco_do_brasil'
    if any(kw in text_lower for kw in ['nubank', 'nu pagamentos', 'nu - pagamentos', 'n u pagamentos']) or text.startswith('NU'):
        return 'nubank'
    if any(kw in text_lower for kw in ['pagbank', 'pagseguro', 'pag bank', 'pag seguro', 'pagseg']):
        return 'pagbank'
    if any(kw in text_lower for kw in ['mercado pago', 'mercadopago', 'mercado libre']):
        return 'mercado_pago'
    if any(kw in text_lower for kw in ['picpay', 'pic pay']):
        return 'picpay'
    if 'cresol' in text_lower:
        return 'cresol'
    if any(kw in text_lower for kw in ['itaú', 'itau', 'itaü', 'ita\u00fa']):
        return 'itau'
    if 'sicredi' in text_lower:
        return 'sicredi'
    if re.search(r'\bc6\b', text_lower) or 'c6 bank' in text_lower or 'bco c6' in text_lower:
        return 'c6'
    if any(kw in text_lower for kw in ['banco inter', 'bco inter', 'inter s.a', 'inter sa']):
        return 'inter'
    if any(kw in text_lower for kw in ['caixa econ', 'caixa eco', 'caixa federal', 'lotérica', 'loterica']):
        return 'caixa'
    if any(kw in text_lower for kw in ['bradesco', 'bco bradesco']):
        return 'bradesco'
    if 'santander' in text_lower:
        return 'santander'
    if 'sicoob' in text_lower:
        return 'sicoob'
    if any(kw in text_lower for kw in ['original', 'banco original']):
        return 'original'
    # Fallback: tenta detectar pelo ISPB da instituição recebedor/pagador
    ispb_map = {
        '60701190': 'itau', '00000000': 'banco_do_brasil', '60746948': 'bradesco',
        '90400888': 'santander', '18236120': 'nubank', '10573521': 'mercado_pago',
        '60900292': 'caixa', '04902979': 'banco_do_brasil', '22896431': 'pagbank',
        '20855875': 'inter', '07689002': 'sicredi', '08561701': 'pagbank',
        '13370835': 'picpay', '31872495': 'c6', '87711670': 'sicredi',
        '02038232': 'sicoob', '60746948': 'bradesco',
    }
    for ispb, bank in ispb_map.items():
        if ispb in text:
            return bank

    return 'desconhecido'


# ============================================================
# PARSERS POR BANCO
# ============================================================

def parse_nubank(text: str) -> PixData:
    """Parser para comprovantes Nubank."""
    data = PixData(banco_origem='nubank', raw_text=text)

    valor, valor_raw = find_valor(text)
    data.valor = valor
    data.valor_raw = valor_raw
    data.data_hora = find_data(text)
    data.id_transacao = find_id_transacao(text)
    data.chave_pix = find_chave_pix(text)

    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # Nubank OCR pode vir em dois formatos:
    # 1) "Nome\nMARIA EDUARDA" (label e valor em linhas separadas)
    # 2) "Nome MARIA EDUARDA CAVATON" (label e valor na mesma linha)
    # Detectar formato checando se "Nome" aparece sozinho ou com valor

    def extract_inline_or_nextline(lines_list: list[str], label: str) -> Optional[str]:
        """Extrai valor que pode estar na mesma linha ou na próxima."""
        for i, line in enumerate(lines_list):
            # Formato inline: "Label valor_aqui"
            inline_match = re.match(rf'^{label}\s+(.+)$', line, re.IGNORECASE)
            if inline_match:
                return inline_match.group(1).strip()
            # Formato separado: "Label\nvalor"
            if re.match(rf'^{label}$', line, re.IGNORECASE) and i + 1 < len(lines_list):
                return lines_list[i + 1].strip()
        return None

    # Find section boundaries
    destino_idx = None
    origem_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^Destino$', line, re.IGNORECASE):
            destino_idx = i
        if re.match(r'^Origem$', line, re.IGNORECASE):
            origem_idx = i

    # Destino (recebedor)
    if destino_idx is not None:
        end = origem_idx if origem_idx else min(destino_idx + 15, len(lines))
        section = lines[destino_idx + 1:end]
        data.nome_recebedor = extract_inline_or_nextline(section, 'Nome')
        data.cpf_recebedor = extract_inline_or_nextline(section, 'CPF')
        data.instituicao_recebedor = extract_inline_or_nextline(section, 'Institui[çc][ãa]o')
        if not data.chave_pix:
            data.chave_pix = extract_inline_or_nextline(section, 'Chave Pix')

    # Origem (pagador)
    if origem_idx is not None:
        section = lines[origem_idx + 1:min(origem_idx + 15, len(lines))]
        data.nome_pagador = extract_inline_or_nextline(section, 'Nome')
        data.cpf_pagador = extract_inline_or_nextline(section, '[Cc][Pp][Ff]')
        data.instituicao_pagador = extract_inline_or_nextline(section, 'Institui[çc][ãa]o')

    return data


def parse_pagbank(text: str) -> PixData:
    """Parser para comprovantes PagBank."""
    data = PixData(banco_origem='pagbank', raw_text=text)

    valor, valor_raw = find_valor(text)
    data.valor = valor
    data.valor_raw = valor_raw
    data.data_hora = find_data(text)
    data.id_transacao = find_id_transacao(text)
    data.chave_pix = find_chave_pix(text)

    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # Encontra marcadores "De" e "Para" (linhas isoladas)
    de_idx = None
    para_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^De$', line, re.IGNORECASE) and de_idx is None:
            de_idx = i
        if re.match(r'^Para$', line, re.IGNORECASE) and para_idx is None:
            para_idx = i

    def _parse_de_para_section(section_lines):
        """Parse De/Para section: first line = nome, depois CPF/Instituição."""
        result = {'nome': None, 'cpf': None, 'instituicao': None}
        if not section_lines:
            return result
        first = section_lines[0]
        if not re.match(r'^(CPF|CNPJ|Institui|Banco|Chave|Dados|Valor)', first, re.IGNORECASE):
            result['nome'] = first
        for i, line in enumerate(section_lines):
            if re.match(r'^(CPF|CNPJ|CPF\s*/?\s*CNP)', line, re.IGNORECASE) and i + 1 < len(section_lines) and not result['cpf']:
                result['cpf'] = section_lines[i + 1].strip()
            if re.match(r'^Institui', line, re.IGNORECASE) and i + 1 < len(section_lines) and not result['instituicao']:
                result['instituicao'] = section_lines[i + 1].strip()
        return result

    if de_idx is not None:
        end = para_idx if para_idx and para_idx > de_idx else min(de_idx + 10, len(lines))
        section = lines[de_idx + 1:end]
        parsed = _parse_de_para_section(section)
        if parsed['nome']:
            data.nome_pagador = parsed['nome']
        if parsed['cpf']:
            data.cpf_pagador = parsed['cpf']
        if parsed['instituicao']:
            data.instituicao_pagador = parsed['instituicao']

    if para_idx is not None:
        end = min(para_idx + 10, len(lines))
        for i in range(para_idx + 1, min(para_idx + 10, len(lines))):
            if i < len(lines) and re.match(r'^(Chave|Para\s+d[uú]vidas)', lines[i], re.IGNORECASE):
                end = i
                break
        section = lines[para_idx + 1:end]
        parsed = _parse_de_para_section(section)
        if parsed['nome']:
            data.nome_recebedor = parsed['nome']
        if parsed['cpf']:
            data.cpf_recebedor = parsed['cpf']
        if parsed['instituicao']:
            data.instituicao_recebedor = parsed['instituicao']

    # ID transação PagBank: "Código da transação Pix" pode ter IDs sem prefixo E
    if not data.id_transacao:
        for i, line in enumerate(lines):
            if re.match(r'^C.digo\s+da\s+transa.+\s+Pix', line, re.IGNORECASE) and i + 1 < len(lines):
                val = lines[i + 1].strip()
                if len(val) >= 20:
                    data.id_transacao = val
                break

    return data


def parse_banco_do_brasil(text: str) -> PixData:
    """Parser para comprovantes Banco do Brasil."""
    data = PixData(banco_origem='banco_do_brasil', raw_text=text)

    valor, valor_raw = find_valor(text)
    data.valor = valor
    data.valor_raw = valor_raw
    data.data_hora = find_data(text)
    data.chave_pix = find_chave_pix(text)

    # ID transação
    id_match = re.search(r'ID[:\s]*\n?(E[\w]{20,})', text)
    if id_match:
        data.id_transacao = id_match.group(1)

    # BB layout: labels and values on separate lines
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # Find Recebedor section
    receb_idx = None
    pag_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^Recebedor$', line, re.IGNORECASE):
            receb_idx = i
        if re.match(r'^Pagador$', line, re.IGNORECASE):
            pag_idx = i

    # Extract recebedor data from line-by-line structure
    if receb_idx is not None:
        end_idx = pag_idx if pag_idx else len(lines)
        section = lines[receb_idx:end_idx]
        # Nome is typically right after "Recebedor"
        for j, line in enumerate(section):
            if j == 0:
                continue  # skip "Recebedor" label
            if line in ['CPF', 'Agência', 'Conta', 'Instituição', 'Tipo de conta', 'Chave Pix']:
                continue
            if not data.nome_recebedor and not re.match(r'^[\*\+\d\.\-]+$', line) and not re.match(r'^\d{4,}', line):
                data.nome_recebedor = line
                break

        # CPF - line after "CPF" label
        for j, line in enumerate(section):
            if line == 'CPF' and j + 1 < len(section):
                data.cpf_recebedor = section[j + 1]
                break

        # Instituição
        for j, line in enumerate(section):
            if line == 'Instituição' and j + 1 < len(section):
                inst = section[j + 1]
                # Remove ISPB code prefix
                inst = re.sub(r'^\d{8}\s+', '', inst)
                data.instituicao_recebedor = inst
                break

    # Extract pagador data
    if pag_idx is not None:
        section = lines[pag_idx:]
        for j, line in enumerate(section):
            if j == 0:
                continue
            if line.lower() in ['cpf', 'cpr', 'agência', 'conta', 'instituição']:
                continue
            if not data.nome_pagador and not re.match(r'^[\*\+\d\.\-\w]{1,15}$', line) and not re.match(r'^\d{4,}', line):
                data.nome_pagador = line
                break

        for j, line in enumerate(section):
            if line.lower() in ['cpf', 'cpr'] and j + 1 < len(section):
                data.cpf_pagador = section[j + 1]
                break

        for j, line in enumerate(section):
            if line == 'Instituição' and j + 1 < len(section):
                inst = section[j + 1]
                inst = re.sub(r'^\d{8}\s+', '', inst)
                data.instituicao_pagador = inst
                break

    return data


def parse_mercado_pago(text: str) -> PixData:
    """Parser para comprovantes Mercado Pago."""
    data = PixData(banco_origem='mercado_pago', raw_text=text)

    valor, valor_raw = find_valor(text)
    data.valor = valor
    data.valor_raw = valor_raw
    data.data_hora = find_data(text)

    # ID transação PIX
    id_match = re.search(r'ID\s+de\s+transação\s+PIX\s+([\w]+)', text, re.IGNORECASE)
    if id_match:
        data.id_transacao = id_match.group(1)
    else:
        data.id_transacao = find_id_transacao(text)

    # Mercado Pago: line-by-line parsing since OCR removes R$ from large values
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # Find section markers first (needed for valor fallback)
    de_idx = None
    para_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^[e•o>]?\s*De$', line, re.IGNORECASE):
            de_idx = i
        if re.match(r'^[e•o>]?\s*Para$', line, re.IGNORECASE):
            para_idx = i

    # If valor not found, look for standalone number (OCR issues like "R$ 5O")
    # Also handles cases where Tesseract completely misses the R$ line
    if not data.valor:
        # Strategy 1: look for line with just a number between header and "De"
        de_line = de_idx if de_idx is not None else len(lines)
        for i in range(min(de_line, 8)):
            line = lines[i] if i < len(lines) else ''
            # Skip known non-value lines
            if any(kw in line.lower() for kw in ['transação', 'número', 'id', 'atendimento', 'ouvidoria', 'mercado', 'pago', 'comprovante', 'sábado', 'domingo']):
                continue
            # OCR pode ler "R$ 5O" como texto com O em vez de 0
            cleaned = re.sub(r'[Oo]', '0', line)
            cleaned = re.sub(r'[Il]', '1', cleaned)
            v, vr = find_valor(cleaned)
            if v and v < 100000:
                data.valor = v
                data.valor_raw = line
                break
            # Standalone number like "50" or "5O"
            match = re.match(r'^R?\$?\s*(\d{1,6})$', cleaned.strip())
            if match:
                try:
                    val = float(match.group(1))
                    if 0 < val < 100000:
                        data.valor = val
                        data.valor_raw = line
                        break
                except:
                    pass

    # De section - extract pagador data

    if de_idx is not None:
        end = para_idx if para_idx else min(de_idx + 8, len(lines))
        section = lines[de_idx + 1:end]
        # First non-label line is the name
        for line in section:
            if not re.match(r'^(CPF|Agência|Número|Mercado|\d)', line, re.IGNORECASE):
                data.nome_pagador = line
                break
        # CPF
        for line in section:
            cpf_m = re.search(r'CPF[:\s]+([\*\+\.\d\-\s]+)', line)
            if cpf_m:
                data.cpf_pagador = cpf_m.group(1).strip()
                break
        data.instituicao_pagador = 'Mercado Pago'

    if para_idx is not None:
        section = lines[para_idx + 1:para_idx + 8]
        for line in section:
            if not re.match(r'^(CPF|BANCO|Agência|Número|\d)', line, re.IGNORECASE):
                data.nome_recebedor = line
                break
        for line in section:
            cpf_m = re.search(r'CPF[:\s]+([\*\+\.\d\-\s]+)', line)
            if cpf_m:
                data.cpf_recebedor = cpf_m.group(1).strip()
                break
        for line in section:
            if re.match(r'BANCO', line, re.IGNORECASE):
                data.instituicao_recebedor = line.strip()
                break

    data.chave_pix = find_chave_pix(text)

    return data


def parse_cresol(text: str) -> PixData:
    """Parser para comprovantes Cresol."""
    data = PixData(banco_origem='cresol', raw_text=text)

    valor, valor_raw = find_valor(text)
    data.valor = valor
    data.valor_raw = valor_raw
    data.id_transacao = find_id_transacao(text)
    data.chave_pix = find_chave_pix(text)

    # Cresol has sections: Dados do recebedor, Dados da transação, Dados do pagador
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # Find section markers
    receb_start = None
    trans_start = None
    pag_start = None
    for i, line in enumerate(lines):
        if 'recebedor' in line.lower() and 'dados' in line.lower():
            receb_start = i
        if 'transação' in line.lower() and 'dados' in line.lower():
            trans_start = i
        if 'pagador' in line.lower() and 'dados' in line.lower():
            pag_start = i

    def parse_cresol_section(start_idx, end_idx):
        """Parse Nome/CPF/Instituição from a Cresol section."""
        section = lines[start_idx:end_idx] if end_idx else lines[start_idx:]
        nome = None
        cpf = None
        inst = None
        for j, line in enumerate(section):
            if line == 'Nome' and j + 1 < len(section):
                nome = section[j + 1]
            if line.startswith('CPF') and j + 1 < len(section):
                cpf = section[j + 1].strip().strip('"').strip("'")
            if line == 'Instituição' and j + 1 < len(section):
                inst = section[j + 1]
        return nome, cpf, inst

    if receb_start is not None:
        end = trans_start or pag_start or len(lines)
        nome, cpf, inst = parse_cresol_section(receb_start, end)
        data.nome_recebedor = nome
        data.cpf_recebedor = cpf
        data.instituicao_recebedor = inst

    if pag_start is not None:
        nome, cpf, inst = parse_cresol_section(pag_start, None)
        data.nome_pagador = nome
        data.cpf_pagador = cpf
        data.instituicao_pagador = inst

    # Data/hora from transação section
    if trans_start is not None:
        for i in range(trans_start, min(trans_start + 6, len(lines))):
            line = lines[i]
            if 'Data' in line and 'hora' in line and i + 1 < len(lines):
                data.data_hora = lines[i + 1]
                break
            date_match = re.search(r'(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})', line)
            if date_match:
                data.data_hora = date_match.group(1)
                break

    # Fallback for date
    if not data.data_hora:
        data.data_hora = find_data(text)

    return data


def parse_itau(text: str) -> PixData:
    """Parser para comprovantes Itaú."""
    data = PixData(banco_origem='itau', raw_text=text)

    valor, valor_raw = find_valor(text)
    data.valor = valor
    data.valor_raw = valor_raw
    data.data_hora = find_data(text)
    data.chave_pix = find_chave_pix(text)

    # ID transação
    id_match = re.search(r'ID\s+da\s+transação[:\s]*(E[\w]+)', text, re.IGNORECASE)
    if id_match:
        data.id_transacao = id_match.group(1)
    else:
        data.id_transacao = find_id_transacao(text)

    # De
    de_match = re.search(
        r'\bDe\b\s+([A-ZÀ-Ú][A-ZÀ-Ú\s]+)\s+CPF[:\s]+([\*\.\d\-]+).*?Institui[çc][ãa]o[:\s]+(.*?)(?:\n\n|\bPara\b)',
        text, re.DOTALL
    )
    if de_match:
        data.nome_pagador = de_match.group(1).strip()
        data.cpf_pagador = de_match.group(2).strip()
        data.instituicao_pagador = de_match.group(3).strip()

    # Para
    para_match = re.search(
        r'\bPara\b\s+([A-ZÀ-Ú][A-ZÀ-Ú\s]+)\s+CPF[:\s]+([\*\.\d\-]+).*?Institui[çc][ãa]o[:\s]+(.*?)(?:\n|Chave)',
        text, re.DOTALL
    )
    if para_match:
        data.nome_recebedor = para_match.group(1).strip()
        data.cpf_recebedor = para_match.group(2).strip()
        data.instituicao_recebedor = para_match.group(3).strip()

    return data


def parse_caixa(text: str) -> PixData:
    """Parser para comprovantes Caixa (inclui Mega da Virada etc)."""
    data = PixData(banco_origem='caixa', raw_text=text)

    valor, valor_raw = find_valor(text)
    data.valor = valor
    data.valor_raw = valor_raw
    data.data_hora = find_data(text)

    # Detecta se é aposta ou PIX
    if 'aposta' in text.lower() or 'mega' in text.lower():
        data.tipo = 'aposta_loteria'

    # Line-by-line parsing
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # Find Nome label
    for i, line in enumerate(lines):
        if line == 'Nome' and i + 1 < len(lines):
            data.nome_pagador = lines[i + 1]
            break

    # Fallback: Pagador section
    if not data.nome_pagador:
        for i, line in enumerate(lines):
            if 'Pagador' in line and i + 1 < len(lines):
                # Next line might be "Nome", skip to value
                for j in range(i + 1, min(i + 4, len(lines))):
                    if lines[j] == 'Nome' and j + 1 < len(lines):
                        data.nome_pagador = lines[j + 1]
                        break
                    elif re.match(r'^[A-ZÀ-Ú][A-ZÀ-Ú\s]{3,}$', lines[j]):
                        data.nome_pagador = lines[j]
                        break
                break

    # CPF - line after "CPF" label
    for i, line in enumerate(lines):
        if line == 'CPF' and i + 1 < len(lines):
            data.cpf_pagador = lines[i + 1]
            break

    data.instituicao_pagador = 'CAIXA'

    # Código da operação como ID
    cod_match = re.search(r'Código\s+da\s+Operação\s+(\d+)', text, re.IGNORECASE)
    if cod_match:
        data.id_transacao = cod_match.group(1)

    # For loteria, there's no recebedor (it's CAIXA itself)
    if data.tipo == 'aposta_loteria':
        data.nome_recebedor = 'CAIXA ECONÔMICA FEDERAL'
        data.instituicao_recebedor = 'CAIXA'

    return data


def _parse_section_line_by_line(lines: list[str]) -> dict:
    """Extrai Nome, CPF/CNPJ, Instituição/Banco de uma seção linha a linha."""
    result = {'nome': None, 'cpf': None, 'instituicao': None}
    for i, line in enumerate(lines):
        # Nome: label "NOME" ou "Para" ou "De" seguido do valor na próxima linha
        if re.match(r'^(NOME|Para|De)$', line, re.IGNORECASE) and i + 1 < len(lines) and not result['nome']:
            val = lines[i + 1].strip()
            if val and not re.match(r'^(CPF|CNPJ|CNP|BANCO|CHAVE|DADOS|INSTITUIÇÃO|Para|De)', val, re.IGNORECASE):
                result['nome'] = val
        # CPF/CNPJ: label seguido do valor na próxima linha
        if re.match(r'^(CPF|CNPJ|CPF\s*/?\s*CNP\W*J?|CNP\W*J?)$', line, re.IGNORECASE) and i + 1 < len(lines):
            val = lines[i + 1].strip()
            if val and not re.match(r'^(NOME|BANCO|CHAVE|DADOS|INSTITUIÇÃO)', val, re.IGNORECASE):
                result['cpf'] = val
        # Instituição/Banco
        if re.match(r'^(INSTITUIÇÃO|INSTITUI|BANCO)$', line, re.IGNORECASE) and i + 1 < len(lines):
            val = lines[i + 1].strip()
            if val and not re.match(r'^(NOME|CPF|CNPJ|CHAVE|DADOS)', val, re.IGNORECASE):
                result['instituicao'] = val
        # Banco inline: "BANCO <nome>"
        if re.match(r'^BANCO\s+\w', line, re.IGNORECASE) and not result['instituicao']:
            result['instituicao'] = line.strip()
    return result


def parse_generic(text: str) -> PixData:
    """Parser genérico — tenta extrair o máximo possível."""
    data = PixData(banco_origem='desconhecido', raw_text=text)

    valor, valor_raw = find_valor(text)
    data.valor = valor
    data.valor_raw = valor_raw
    data.data_hora = find_data(text)
    data.id_transacao = find_id_transacao(text)
    data.chave_pix = find_chave_pix(text)

    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # ---- Tenta layout estruturado por seções ----
    # Detecta seções "DADOS DO PAGADOR", "DADOS DO FAVORECIDO", "DADOS DA TRANSFERÊNCIA", etc.
    section_markers = []
    for i, line in enumerate(lines):
        if re.match(r'^DADOS\s+D[OAE]\s+(PAGADOR|FAVORECIDO|RECEBEDOR|TRANSFER)', line, re.IGNORECASE):
            section_markers.append((i, line))

    if section_markers:
        # Layout estruturado encontrado — parseia cada seção
        for idx, (start, marker) in enumerate(section_markers):
            end = section_markers[idx + 1][0] if idx + 1 < len(section_markers) else len(lines)
            section_lines = lines[start + 1:end]
            parsed = _parse_section_line_by_line(section_lines)

            marker_lower = marker.lower()
            if 'pagador' in marker_lower:
                if parsed['nome'] and not data.nome_pagador:
                    data.nome_pagador = parsed['nome']
                if parsed['cpf'] and not data.cpf_pagador:
                    data.cpf_pagador = parsed['cpf']
                if parsed['instituicao'] and not data.instituicao_pagador:
                    data.instituicao_pagador = parsed['instituicao']
            elif 'favorecido' in marker_lower or 'recebedor' in marker_lower or 'transfer' in marker_lower:
                if parsed['nome'] and not data.nome_recebedor:
                    data.nome_recebedor = parsed['nome']
                if parsed['cpf'] and not data.cpf_recebedor:
                    data.cpf_recebedor = parsed['cpf']
                if parsed['instituicao'] and not data.instituicao_recebedor:
                    data.instituicao_recebedor = parsed['instituicao']
                # Chave PIX pode estar nesta seção
                if not data.chave_pix:
                    for sl in section_lines:
                        if re.match(r'^CHAVE$', sl, re.IGNORECASE):
                            ci = section_lines.index(sl)
                            if ci + 1 < len(section_lines):
                                data.chave_pix = section_lines[ci + 1].strip()
                                break

        # Se "DADOS DA TRANSFERÊNCIA" trouxe recebedor, mas não tem seção "FAVORECIDO", ok
        # Data/hora pode estar em seção separada "DATA E HORA..."
        if not data.data_hora:
            for i, line in enumerate(lines):
                if re.match(r'^DATA\s+E\s+HORA', line, re.IGNORECASE) and i + 1 < len(lines):
                    data.data_hora = find_data(lines[i + 1])
                    if not data.data_hora:
                        data.data_hora = lines[i + 1].strip()
                    break

        # ID da transação pode estar após "ID DA TRANSAÇÃO"
        if not data.id_transacao:
            for i, line in enumerate(lines):
                if re.match(r'^ID\s+DA\s+TRANSA', line, re.IGNORECASE) and i + 1 < len(lines):
                    val = lines[i + 1].strip()
                    if val.startswith('E'):
                        data.id_transacao = val
                    break

        return data

    # ---- C6-style: "Banco: CODE - NAME" + "Conta de origem" layout ----
    conta_origem_idx = None
    banco_lines = []
    for i, line in enumerate(lines):
        if re.match(r'^Conta\s+de\s+origem', line, re.IGNORECASE):
            conta_origem_idx = i
        m = re.match(r'^Banco:\s*\d+\s*-\s*(.+)', line, re.IGNORECASE)
        if m:
            banco_lines.append((i, m.group(1).strip()))

    if conta_origem_idx is not None and banco_lines:
        # Recebedor: primeiro "Banco:" antes de "Conta de origem"
        for idx, inst in banco_lines:
            if idx < conta_origem_idx:
                data.instituicao_recebedor = inst
                if idx > 0:
                    candidate = lines[idx - 1]
                    if len(candidate) > 2 and not re.match(r'^(Banco|Ag[eê]ncia|Conta|CPF|CNPJ|Chave|Valor|ID|C.digo|Data|Pix)', candidate, re.IGNORECASE):
                        data.nome_recebedor = candidate
                break
        # Pagador: primeiro "Banco:" depois de "Conta de origem"
        for idx, inst in banco_lines:
            if idx > conta_origem_idx:
                data.instituicao_pagador = inst
                for j in range(idx - 1, conta_origem_idx, -1):
                    candidate = lines[j]
                    if len(candidate) > 2 and not re.match(r'^(Banco|Ag[eê]ncia|Conta|CPF|CNPJ|Chave|Valor|ID|C.digo|Data|Pix)', candidate, re.IGNORECASE):
                        data.nome_pagador = candidate
                        break
                break
        # CPF / CNPJ
        for i, line in enumerate(lines):
            if re.match(r'^(CPF|CNPJ|CPF\s*/?\s*CNP)', line, re.IGNORECASE) and i + 1 < len(lines):
                cpf_val = lines[i + 1].strip()
                if i < conta_origem_idx:
                    if not data.cpf_recebedor:
                        data.cpf_recebedor = cpf_val
                else:
                    if not data.cpf_pagador:
                        data.cpf_pagador = cpf_val

        return data

    # ---- De/Para standalone section layout ----
    de_idx = None
    para_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^De$', line, re.IGNORECASE) and de_idx is None:
            de_idx = i
        if re.match(r'^Para$', line, re.IGNORECASE) and para_idx is None:
            para_idx = i

    if de_idx is not None or para_idx is not None:
        def _parse_de_para(section_lines):
            result = {'nome': None, 'cpf': None, 'instituicao': None}
            if not section_lines:
                return result
            first = section_lines[0]
            if not re.match(r'^(CPF|CNPJ|Institui|Banco|Chave|Dados|Valor)', first, re.IGNORECASE):
                result['nome'] = first
            for i, line in enumerate(section_lines):
                if re.match(r'^(CPF|CNPJ|CPF\s*/?\s*CNP)', line, re.IGNORECASE) and i + 1 < len(section_lines) and not result['cpf']:
                    result['cpf'] = section_lines[i + 1].strip()
                if re.match(r'^Institui', line, re.IGNORECASE) and i + 1 < len(section_lines) and not result['instituicao']:
                    result['instituicao'] = section_lines[i + 1].strip()
            return result

        if de_idx is not None:
            end = para_idx if para_idx and para_idx > de_idx else min(de_idx + 10, len(lines))
            section = lines[de_idx + 1:end]
            parsed = _parse_de_para(section)
            if parsed['nome'] and not data.nome_pagador:
                data.nome_pagador = parsed['nome']
            if parsed['cpf'] and not data.cpf_pagador:
                data.cpf_pagador = parsed['cpf']
            if parsed['instituicao'] and not data.instituicao_pagador:
                data.instituicao_pagador = parsed['instituicao']

        if para_idx is not None:
            end = min(para_idx + 10, len(lines))
            for i in range(para_idx + 1, min(para_idx + 10, len(lines))):
                if i < len(lines) and re.match(r'^(Chave|Para\s+d[uú]vidas)', lines[i], re.IGNORECASE):
                    end = i
                    break
            section = lines[para_idx + 1:end]
            parsed = _parse_de_para(section)
            if parsed['nome'] and not data.nome_recebedor:
                data.nome_recebedor = parsed['nome']
            if parsed['cpf'] and not data.cpf_recebedor:
                data.cpf_recebedor = parsed['cpf']
            if parsed['instituicao'] and not data.instituicao_recebedor:
                data.instituicao_recebedor = parsed['instituicao']

        return data

    # ---- Fallback: "Quem recebeu" / "Quem pagou" layout (Neon, etc.) ----
    def _extract_inline_labels(section_lines: list[str]) -> dict:
        """Extrai Nome, CPF, Instituição de linhas com labels inline ou em linhas separadas."""
        result = {'nome': None, 'cpf': None, 'instituicao': None}
        for i, line in enumerate(section_lines):
            # Nome inline: "Nome Maria Eduarda"
            m = re.match(r'^Nome\s+(.+)', line, re.IGNORECASE)
            if m and not result['nome']:
                nome = m.group(1).strip()
                # Se a próxima linha parece continuação do nome (sem label)
                if i + 1 < len(section_lines):
                    next_l = section_lines[i + 1]
                    if not re.match(r'^(CPF|CNPJ|CNP|Institui|Banco|Chave|ID|Dados|Quem|Nome|Ag[eê]ncia|Conta|Tipo)', next_l, re.IGNORECASE):
                        nome = nome + ' ' + next_l.strip()
                result['nome'] = nome
            # Nome em linha separada: "Nome\nMARIA EDUARDA"
            if re.match(r'^Nome$', line, re.IGNORECASE) and i + 1 < len(section_lines) and not result['nome']:
                val = section_lines[i + 1].strip()
                if val and not re.match(r'^(CPF|CNPJ|CNP|Institui|Banco|Chave|Dados)', val, re.IGNORECASE):
                    # Verificar continuação na próxima linha
                    if i + 2 < len(section_lines):
                        next_l = section_lines[i + 2]
                        if not re.match(r'^(CPF|CNPJ|CNP|Institui|Banco|Chave|ID|Dados|Quem|Nome|Ag[eê]ncia|Conta|Tipo)', next_l, re.IGNORECASE):
                            val = val + ' ' + next_l.strip()
                    result['nome'] = val
            # CPF inline: "CPF / CNPJ ***824,458-**" ou "CPF ***824,458-**"
            m = re.match(r'^(?:CPF\s*/?\s*CNP\W*J?|CPF|CNPJ)[\s:]+(.+)', line, re.IGNORECASE)
            if m and not result['cpf']:
                result['cpf'] = m.group(1).strip()
            # CPF em linha separada: "CPF\n* B24.458-**"
            if re.match(r'^(?:CPF\s*/?\s*CNP\W*J?|CPF|CNPJ)$', line, re.IGNORECASE) and i + 1 < len(section_lines) and not result['cpf']:
                val = section_lines[i + 1].strip()
                if val and not re.match(r'^(Nome|Institui|Banco|Chave|Dados|Quem|Ag[eê]ncia|Conta)', val, re.IGNORECASE):
                    result['cpf'] = val
            # Instituição inline: "Instituição BANCO INTER"
            m = re.match(r'^Institui\S*\s+(.+)', line, re.IGNORECASE)
            if m and not result['instituicao']:
                result['instituicao'] = m.group(1).strip()
            # Banco em linha separada: "Banco\n00416968 - BANCO INTER"
            if re.match(r'^Banco$', line, re.IGNORECASE) and i + 1 < len(section_lines) and not result['instituicao']:
                val = section_lines[i + 1].strip()
                if val and not re.match(r'^(Nome|CPF|CNPJ|Chave|Dados|Quem)', val, re.IGNORECASE):
                    # Extrai nome do banco após "CÓDIGO - "
                    m2 = re.match(r'[\d\s]+\s*-\s*(.+)', val)
                    if m2:
                        result['instituicao'] = m2.group(1).strip()
                    else:
                        result['instituicao'] = val
        return result

    # ---- Destino/Origem section layout (inline labels) ----
    destino_idx_g = None
    origem_idx_g = None
    for i, line in enumerate(lines):
        if re.match(r'^Destino$', line, re.IGNORECASE):
            destino_idx_g = i
        if re.match(r'^Origem$', line, re.IGNORECASE):
            origem_idx_g = i

    if destino_idx_g is not None:
        end = origem_idx_g if origem_idx_g and origem_idx_g > destino_idx_g else min(destino_idx_g + 15, len(lines))
        section = lines[destino_idx_g + 1:end]
        parsed = _extract_inline_labels(section)
        if parsed['nome']:
            data.nome_recebedor = parsed['nome']
        if parsed['cpf']:
            data.cpf_recebedor = parsed['cpf']
        if parsed['instituicao']:
            data.instituicao_recebedor = parsed['instituicao']
        if not data.chave_pix:
            for sl in section:
                m = re.match(r'^Chave\s+Pix\s+(.+)', sl, re.IGNORECASE)
                if m:
                    data.chave_pix = m.group(1).strip()
                    break
                if re.match(r'^Chave\s+Pix$', sl, re.IGNORECASE):
                    si = section.index(sl)
                    if si + 1 < len(section):
                        data.chave_pix = section[si + 1].strip()
                    break

    if origem_idx_g is not None:
        if destino_idx_g and destino_idx_g > origem_idx_g:
            section = lines[origem_idx_g + 1:destino_idx_g]
        else:
            section = lines[origem_idx_g + 1:min(origem_idx_g + 15, len(lines))]
        parsed = _extract_inline_labels(section)
        if parsed['nome']:
            data.nome_pagador = parsed['nome']
        if parsed['cpf']:
            data.cpf_pagador = parsed['cpf']
        if parsed['instituicao']:
            data.instituicao_pagador = parsed['instituicao']

    if destino_idx_g is not None or origem_idx_g is not None:
        return data

    quem_recebeu_idx = None
    quem_pagou_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^Quem\s+recebeu', line, re.IGNORECASE):
            quem_recebeu_idx = i
        if re.match(r'^Quem\s+pagou', line, re.IGNORECASE):
            quem_pagou_idx = i

    if quem_recebeu_idx is not None:
        end = quem_pagou_idx if quem_pagou_idx and quem_pagou_idx > quem_recebeu_idx else min(quem_recebeu_idx + 12, len(lines))
        section = lines[quem_recebeu_idx + 1:end]
        parsed = _extract_inline_labels(section)
        if parsed['nome']:
            data.nome_recebedor = parsed['nome']
        if parsed['cpf']:
            data.cpf_recebedor = parsed['cpf']
        if parsed['instituicao']:
            data.instituicao_recebedor = parsed['instituicao']

    if quem_pagou_idx is not None:
        end = quem_recebeu_idx if quem_recebeu_idx and quem_recebeu_idx > quem_pagou_idx else min(quem_pagou_idx + 12, len(lines))
        section = lines[quem_pagou_idx + 1:end]
        parsed = _extract_inline_labels(section)
        if parsed['nome']:
            data.nome_pagador = parsed['nome']
        if parsed['cpf']:
            data.cpf_pagador = parsed['cpf']
        if parsed['instituicao']:
            data.instituicao_pagador = parsed['instituicao']

    # Instituição antes de "Quem recebeu" pode ser do pagador (Neon)
    if not data.instituicao_pagador and quem_recebeu_idx is not None:
        for i in range(max(0, quem_recebeu_idx - 3), quem_recebeu_idx):
            m = re.match(r'^Institui\S*\s+(.+)', lines[i], re.IGNORECASE)
            if m:
                data.instituicao_pagador = m.group(1).strip()
                break

    if quem_recebeu_idx is not None or quem_pagou_idx is not None:
        return data

    # ---- Sicredi-style: repeated Nome/CPF/Banco label groups without section markers ----
    nome_indices = [i for i, line in enumerate(lines) if re.match(r'^Nome$', line, re.IGNORECASE)]
    if len(nome_indices) >= 2:
        # First Nome group = recebedor, second = pagador
        end_first = nome_indices[1]
        first_section = lines[nome_indices[0]:end_first]
        end_second = nome_indices[2] if len(nome_indices) > 2 else min(nome_indices[1] + 10, len(lines))
        second_section = lines[nome_indices[1]:end_second]

        first = _extract_inline_labels(first_section)
        second = _extract_inline_labels(second_section)

        if first['nome']:
            data.nome_recebedor = first['nome']
        if first['cpf']:
            data.cpf_recebedor = first['cpf']
        if first['instituicao']:
            data.instituicao_recebedor = first['instituicao']
        if second['nome']:
            data.nome_pagador = second['nome']
        if second['cpf']:
            data.cpf_pagador = second['cpf']
        if second['instituicao']:
            data.instituicao_pagador = second['instituicao']

        return data

    # ---- Fallback: layout não-estruturado (De/Para, Pagador/Recebedor) ----
    for label in [r'\bDe\b', r'Pagador', r'Origem']:
        match = re.search(
            rf'{label}\s+([A-ZÀ-Úa-zà-ú][A-ZÀ-Úa-zà-ú\s\.]+)',
            text
        )
        if match:
            nome = match.group(1).strip()
            for sw in ['CPF', 'CNPJ', 'Instituição', 'Agência']:
                idx = nome.find(sw)
                if idx > 0:
                    nome = nome[:idx].strip()
            if len(nome) > 2:
                data.nome_pagador = nome
                break

    for label in [r'\bPara\b', r'Recebedor', r'Destino']:
        match = re.search(
            rf'{label}\s+([A-ZÀ-Úa-zà-ú][A-ZÀ-Úa-zà-ú\s\.]+)',
            text
        )
        if match:
            nome = match.group(1).strip()
            for sw in ['CPF', 'CNPJ', 'Instituição', 'Agência']:
                idx = nome.find(sw)
                if idx > 0:
                    nome = nome[:idx].strip()
            if len(nome) > 2:
                data.nome_recebedor = nome
                break

    # CPFs — pega os dois primeiros que encontrar
    cpfs = re.findall(r'[\*\.,]*\s*\d{3}[\.,\s]?\d{3}[\-\.,]\w{2,}', text)
    if len(cpfs) >= 1:
        data.cpf_pagador = cpfs[0] if not data.cpf_pagador else data.cpf_pagador
    if len(cpfs) >= 2:
        data.cpf_recebedor = cpfs[1] if not data.cpf_recebedor else data.cpf_recebedor

    # Instituições
    inst_patterns = [
        r'(BANCO\s+[\w\s\.]+)',
        r'(NU\s+PAGAMENTOS[\w\s\.\-]*)',
        r'(COOP\s+[\w\s]+)',
        r'(Mercado\s+Pago)',
        r'(PagBank[\w\s\.\(\)]*)',
        r'(ITAÚ[\w\s\.]*)',
    ]
    institutions = []
    for pattern in inst_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        institutions.extend(matches)

    if len(institutions) >= 1:
        data.instituicao_recebedor = institutions[0].strip()
    if len(institutions) >= 2:
        data.instituicao_pagador = institutions[1].strip()

    return data


def parse_picpay(text: str) -> PixData:
    """Parser para comprovantes PicPay."""
    data = PixData(banco_origem='picpay', raw_text=text)

    valor, valor_raw = find_valor(text)
    data.valor = valor
    data.valor_raw = valor_raw
    data.data_hora = find_data(text)
    data.id_transacao = find_id_transacao(text)
    data.chave_pix = find_chave_pix(text)

    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # PicPay layout: Para / De sections, nomes podem estar em 2 linhas
    para_idx = None
    de_idx = None
    id_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^Para$', line, re.IGNORECASE) and para_idx is None:
            para_idx = i
        if re.match(r'^De$', line, re.IGNORECASE) and de_idx is None:
            de_idx = i
        if re.match(r'^ID\s+da\s+transa', line, re.IGNORECASE):
            id_idx = i

    def _parse_picpay_section(start, end):
        """Extract nome, CPF, instituicao from a PicPay section."""
        section = lines[start + 1:end] if end else lines[start + 1:start + 8]
        nome = None
        cpf = None
        inst = None
        nome_lines = []
        for idx_s, line in enumerate(section):
            # CPF mascarado: ***.824.458-** (OCR pode corromper para +IHB24.458-**)
            # Heurística: linha curta, sem espaços, com dígitos e chars especiais
            if (' ' not in line and len(line) <= 25
                    and sum(1 for c in line if c.isdigit()) >= 2
                    and re.search(r'[\*\.\-/\+]', line)):
                cpf = line
                # Instituição: próxima linha após CPF
                if idx_s + 1 < len(section):
                    candidate = section[idx_s + 1]
                    if not re.match(r'^(De|Para|ID|Chave|\*)', candidate, re.IGNORECASE):
                        inst = candidate
                break
            # Instituição ou label — para de coletar nome
            if re.match(r'^(ID|Chave|CNPJ|Ouvidoria|SAC|Canal)', line, re.IGNORECASE):
                break
            nome_lines.append(line)
        if nome_lines:
            nome = ' '.join(nome_lines)
        return nome, cpf, inst

    if para_idx is not None:
        end = de_idx if de_idx and de_idx > para_idx else (id_idx or None)
        nome, cpf, inst = _parse_picpay_section(para_idx, end)
        data.nome_recebedor = nome
        data.cpf_recebedor = cpf
        data.instituicao_recebedor = inst

    if de_idx is not None:
        end = id_idx if id_idx and id_idx > de_idx else None
        nome, cpf, inst = _parse_picpay_section(de_idx, end)
        data.nome_pagador = nome
        data.cpf_pagador = cpf
        data.instituicao_pagador = inst

    # Chave Pix: pode estar após "Chave Pix do recebedor"
    if not data.chave_pix:
        for i, line in enumerate(lines):
            if re.match(r'^Chave\s+Pix', line, re.IGNORECASE) and i + 1 < len(lines):
                val = lines[i + 1].strip()
                if val.startswith('+') or re.match(r'^\d{10,}', val):
                    data.chave_pix = val
                break

    return data


# ============================================================
# ROUTER DE PARSERS
# ============================================================

BANK_PARSERS = {
    'nubank': parse_nubank,
    'neon': parse_generic,
    'pagbank': parse_pagbank,
    'banco_do_brasil': parse_banco_do_brasil,
    'mercado_pago': parse_mercado_pago,
    'picpay': parse_picpay,
    'cresol': parse_cresol,
    'itau': parse_itau,
    'caixa': parse_caixa,
    'c6': parse_generic,
    'santander': parse_generic,
}


def parse_receipt(text: str) -> PixData:
    """Classifica o banco e aplica o parser correto."""
    text = clean_text(text)
    bank = classify_bank(text)
    parser = BANK_PARSERS.get(bank, parse_generic)
    data = parser(text)

    # Set banco_origem from classify_bank if parser left it as 'desconhecido'
    if data.banco_origem == 'desconhecido' and bank != 'desconhecido':
        data.banco_origem = bank

    # Limpa campos — remove \n, espaços extras
    for field in ['nome_pagador', 'nome_recebedor', 'cpf_pagador', 'cpf_recebedor',
                  'instituicao_pagador', 'instituicao_recebedor', 'chave_pix', 'id_transacao']:
        val = getattr(data, field, None)
        if val:
            val = re.sub(r'\s+', ' ', val).strip()
            setattr(data, field, val)

    return data


# ============================================================
# TRUST SCORE ENGINE
# ============================================================

def parse_data_hora(data_hora_str: str) -> Optional[datetime]:
    """
    Tenta converter a string de data/hora extraída para datetime.
    Suporta múltiplos formatos comuns de comprovantes brasileiros.
    """
    meses = {
        'jan': '01', 'fev': '02', 'mar': '03', 'abr': '04',
        'mai': '05', 'jun': '06', 'jul': '07', 'ago': '08',
        'set': '09', 'out': '10', 'nov': '11', 'dez': '12',
        'janeiro': '01', 'fevereiro': '02', 'março': '03', 'abril': '04',
        'maio': '05', 'junho': '06', 'julho': '07', 'agosto': '08',
        'setembro': '09', 'outubro': '10', 'novembro': '11', 'dezembro': '12',
    }

    formatos = [
        r'(\d{2})/(\d{2})/(\d{4})\s+(?:às\s+)?(\d{2}):(\d{2}):(\d{2})',
        r'(\d{2})/(\d{2})/(\d{4})\s+(\d{2}):(\d{2})',
        r'(\d{2})/(\d{2})/(\d{4})',
    ]

    for fmt in formatos:
        m = re.search(fmt, data_hora_str)
        if m:
            groups = m.groups()
            try:
                dia, mes, ano = int(groups[0]), int(groups[1]), int(groups[2])
                hora = int(groups[3]) if len(groups) > 3 else 0
                minuto = int(groups[4]) if len(groups) > 4 else 0
                segundo = int(groups[5]) if len(groups) > 5 else 0
                return datetime(ano, mes, dia, hora, minuto, segundo, tzinfo=BRT)
            except (ValueError, IndexError):
                continue

    # Formato: "07 DEZ 2025 - 07:02:21" ou "02/mar/2026 - 19:44:50"
    m = re.search(r'(\d{2})\s+(\w{3,})\s+(\d{4})\s*[-–]\s*(\d{2}):(\d{2}):(\d{2})', data_hora_str)
    if not m:
        m = re.search(r'(\d{2})/(\w{3,})/(\d{4})\s*[-–]\s*(\d{2}):(\d{2}):(\d{2})', data_hora_str)
    if m:
        dia, mes_str, ano = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        hora, minuto, segundo = int(m.group(4)), int(m.group(5)), int(m.group(6))
        mes_num = meses.get(mes_str[:3])
        if mes_num:
            try:
                return datetime(ano, int(mes_num), dia, hora, minuto, segundo, tzinfo=BRT)
            except ValueError:
                pass

    # Formato: "sábado, 18 de março de 2026, às 14:30:00"
    m = re.search(r'(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4}),?\s+às\s+(\d{2}):(\d{2}):(\d{2})', data_hora_str)
    if m:
        dia, mes_str, ano = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        hora, minuto, segundo = int(m.group(4)), int(m.group(5)), int(m.group(6))
        mes_num = meses.get(mes_str[:3])
        if mes_num:
            try:
                return datetime(ano, int(mes_num), dia, hora, minuto, segundo, tzinfo=BRT)
            except ValueError:
                pass

    return None


def is_data_dentro_validade(data_hora_str: str, max_horas: int = MAX_HORAS_VALIDADE) -> tuple[bool, Optional[str]]:
    """
    Verifica se a data/hora do comprovante está dentro da janela de validade.
    Retorna (valido, motivo).
    """
    dt = parse_data_hora(data_hora_str)
    if dt is None:
        return False, "Não foi possível interpretar a data/hora do comprovante"

    agora = datetime.now(BRT)
    diferenca = agora - dt
    horas_diferenca = diferenca.total_seconds() / 3600

    if horas_diferenca < -1:  # 1h de tolerância para fuso/relógio
        return False, f"Data do comprovante está no futuro ({data_hora_str})"

    if horas_diferenca > max_horas:
        return False, f"Comprovante com mais de {max_horas}h ({horas_diferenca:.1f}h atrás)"

    return True, None


def _cpf_fuzzy_match(cpf_extraido: str, expected_partial: str) -> tuple[bool, str]:
    """
    Matching fuzzy de CPF tolerante a erros de OCR.
    Retorna (match, nivel) onde nivel é 'exato', 'fuzzy' ou 'falha'.
    """
    if not cpf_extraido or not expected_partial:
        return False, 'falha'
    
    cpf_limpo = cpf_extraido.replace(' ', '')
    expected_digits = re.sub(r'[^\d]', '', expected_partial)
    cpf_digits = re.sub(r'[^\d]', '', cpf_limpo)
    
    # Match exato de substring
    if expected_partial in cpf_limpo or expected_digits in cpf_digits:
        return True, 'exato'
    if len(cpf_digits) >= 3 and cpf_digits in expected_digits:
        return True, 'exato'
    
    # Match fuzzy: OCR troca dígitos (4→A, 8→B, 0→O, 5→S, etc.)
    # Extrai dígitos do esperado e tenta achar sequência similar no CPF
    if len(expected_digits) >= 4:
        # Remove chars não-alfanuméricos e tenta match digit-by-digit com tolerância
        cpf_clean = re.sub(r'[^a-zA-Z0-9]', '', cpf_extraido.lower())
        expected_clean = expected_digits
        
        # Mapa de chars confundidos pelo OCR
        ocr_digit_map = {
            '0': 'oO', '1': 'lI|', '2': 'zZ', '3': '', '4': 'Aa',
            '5': 'sS', '6': 'bG', '7': '', '8': 'B', '9': 'g',
        }
        
        best_match = 0
        for start in range(max(1, len(cpf_clean) - len(expected_clean) + 1)):
            matches = 0
            for j, exp_digit in enumerate(expected_clean):
                if start + j >= len(cpf_clean):
                    break
                c = cpf_clean[start + j]
                if c == exp_digit:
                    matches += 1
                elif c in ocr_digit_map.get(exp_digit, ''):
                    matches += 0.7  # OCR confusion match
            best_match = max(best_match, matches)
        
        # Se ≥70% dos dígitos batem (permitindo confusão OCR)
        ratio = best_match / len(expected_clean) if expected_clean else 0
        if ratio >= 0.70:
            return True, 'fuzzy'
    
    return False, 'falha'


def calculate_trust_score(data: PixData) -> TrustScore:
    """
    Calcula trust score baseado em regras determinísticas.
    
    Penalidades calibradas:
    - Ausência de campo (falha de OCR): penalidade leve
    - Mismatch ativo (dado errado): penalidade pesada
    """
    score = 1.0
    detalhes = []
    penalidades = []

    # ---- CAMPOS OBRIGATÓRIOS ----

    # Valor
    if data.valor and data.valor > 0:
        detalhes.append(f"✓ Valor encontrado: R$ {data.valor:.2f}")
    else:
        score -= 0.10
        penalidades.append("✗ Valor não encontrado ou zero (-0.10)")

    # Nome recebedor
    if data.nome_recebedor and len(data.nome_recebedor) > 3:
        detalhes.append(f"✓ Nome recebedor: {data.nome_recebedor}")
        if _nomes_correspondem(data.nome_recebedor, EXPECTED_NOME_RECEBEDOR):
            detalhes.append(f"✓ Nome recebedor confere com o esperado ({EXPECTED_NOME_RECEBEDOR})")
        else:
            score -= 0.35
            penalidades.append(f"✗ Nome recebedor '{data.nome_recebedor}' NÃO confere com o esperado '{EXPECTED_NOME_RECEBEDOR}' (-0.35)")
    else:
        score -= 0.08
        penalidades.append("✗ Nome do recebedor não encontrado (-0.08)")

    # Nome pagador
    if data.nome_pagador and len(data.nome_pagador) > 3:
        detalhes.append(f"✓ Nome pagador: {data.nome_pagador}")
    else:
        score -= 0.05
        penalidades.append("✗ Nome do pagador não encontrado (-0.05)")

    # CPF recebedor - com matching fuzzy tolerante a OCR
    if data.cpf_recebedor:
        detalhes.append(f"✓ CPF recebedor: {data.cpf_recebedor}")
        digits = re.findall(r'\d', data.cpf_recebedor)
        if len(digits) < 4:
            score -= 0.03
            penalidades.append("✗ CPF recebedor com formato incomum (-0.03)")
        
        cpf_match, cpf_nivel = _cpf_fuzzy_match(data.cpf_recebedor, EXPECTED_CPF_RECEBEDOR_PARTIAL)
        if cpf_match and cpf_nivel == 'exato':
            detalhes.append(f"✓ CPF recebedor contém trecho esperado ({EXPECTED_CPF_RECEBEDOR_PARTIAL})")
        elif cpf_match and cpf_nivel == 'fuzzy':
            detalhes.append(f"✓ CPF recebedor confere por similaridade (OCR tolerante)")
            score -= 0.03
            penalidades.append("✗ CPF recebedor reconhecido com tolerância OCR (-0.03)")
        else:
            score -= 0.15
            penalidades.append(f"✗ CPF recebedor '{data.cpf_recebedor}' NÃO contém trecho esperado '{EXPECTED_CPF_RECEBEDOR_PARTIAL}' (-0.15)")
    else:
        score -= 0.05
        penalidades.append("✗ CPF do recebedor não encontrado (-0.05)")

    # CPF pagador
    if data.cpf_pagador:
        detalhes.append(f"✓ CPF pagador: {data.cpf_pagador}")
    else:
        score -= 0.03
        penalidades.append("✗ CPF do pagador não encontrado (-0.03)")

    # Data/hora
    if data.data_hora:
        detalhes.append(f"✓ Data/hora: {data.data_hora}")
        data_valida, motivo = is_data_dentro_validade(data.data_hora)
        if data_valida:
            detalhes.append("✓ Data dentro da janela de validade (últimas 24h)")
        else:
            score -= 0.30
            penalidades.append(f"✗ {motivo} (-0.30)")
    else:
        score -= 0.08
        penalidades.append("✗ Data/hora não encontrada (-0.08)")

    # ---- ID DE TRANSAÇÃO PIX ----

    if data.id_transacao:
        detalhes.append(f"✓ ID transação: {data.id_transacao[:30]}...")
        if re.match(r'^E\d{8}', data.id_transacao):
            detalhes.append("✓ ID segue padrão BACEN (E + ISPB)")
        else:
            score -= 0.05
            penalidades.append("✗ ID de transação não segue padrão BACEN (-0.05)")
        
        # Detecção de duplicatas
        if data.id_transacao in _processed_transaction_ids:
            score -= 0.30
            penalidades.append("✗ ID de transação já foi processado anteriormente — possível duplicata (-0.30)")
            detalhes.append("⚠ ID de transação duplicado detectado")
    else:
        score -= 0.10
        penalidades.append("✗ ID de transação PIX não encontrado (-0.10)")

    # ---- BANCO IDENTIFICADO ----

    if data.banco_origem and data.banco_origem != 'desconhecido':
        detalhes.append(f"✓ Banco identificado: {data.banco_origem}")
    else:
        score -= 0.03
        penalidades.append("✗ Banco emissor não identificado (-0.03)")

    # ---- INSTITUIÇÃO ----

    if data.instituicao_recebedor:
        detalhes.append(f"✓ Instituição recebedor: {data.instituicao_recebedor}")
        if EXPECTED_INSTITUICAO_RECEBEDOR.lower() in data.instituicao_recebedor.lower() or data.instituicao_recebedor.lower() in EXPECTED_INSTITUICAO_RECEBEDOR.lower():
            detalhes.append(f"✓ Instituição recebedor confere com o esperado ({EXPECTED_INSTITUICAO_RECEBEDOR})")
        else:
            score -= 0.25
            penalidades.append(f"✗ Instituição recebedor '{data.instituicao_recebedor}' NÃO confere com '{EXPECTED_INSTITUICAO_RECEBEDOR}' (-0.25)")
    else:
        score -= 0.03
        penalidades.append("✗ Instituição do recebedor não encontrada (-0.03)")

    # ---- CHAVE PIX (normalizada) ----

    if data.chave_pix:
        detalhes.append(f"✓ Chave PIX: {data.chave_pix}")
        chave_norm = _normalize_chave_pix(data.chave_pix)
        expected_norm = _normalize_chave_pix(EXPECTED_CHAVE_PIX)
        
        if expected_norm in chave_norm or chave_norm in expected_norm:
            detalhes.append(f"✓ Chave PIX confere com a esperada ({EXPECTED_CHAVE_PIX})")
        elif len(expected_norm) >= 4:
            # Último recurso: últimos 4 dígitos (chave mascarada)
            ultimos4_expected = re.sub(r'[^\d]', '', EXPECTED_CHAVE_PIX)[-4:]
            digits_chave = re.sub(r'[^\d]', '', data.chave_pix)
            if len(digits_chave) >= 4 and digits_chave[-4:] == ultimos4_expected:
                detalhes.append(f"✓ Chave PIX parcial confere (últimos 4 dígitos: {ultimos4_expected})")
            else:
                score -= 0.30
                penalidades.append(f"✗ Chave PIX '{data.chave_pix}' NÃO confere com '{EXPECTED_CHAVE_PIX}' (-0.30)")
        else:
            score -= 0.30
            penalidades.append(f"✗ Chave PIX '{data.chave_pix}' NÃO confere com '{EXPECTED_CHAVE_PIX}' (-0.30)")
    else:
        score -= 0.02
        penalidades.append("✗ Chave PIX não encontrada (-0.02)")

    # ---- VALIDAÇÕES DE CONSISTÊNCIA ----

    if data.tipo and data.tipo == 'pix':
        detalhes.append("✓ Tipo de transação: PIX")
    else:
        score -= 0.10
        penalidades.append(f"✗ Tipo de transação '{data.tipo}' não é PIX (-0.10)")

    if data.raw_text:
        text_lower = data.raw_text.lower()
        if 'pix' not in text_lower and 'transferência' not in text_lower and 'transferencia' not in text_lower and 'pagamento' not in text_lower:
            score -= 0.05
            penalidades.append("✗ Texto não menciona 'PIX', 'transferência' ou 'pagamento' (-0.05)")

    # ---- DETECÇÃO DE AGENDAMENTO ----
    is_agendado = False
    if data.raw_text:
        text_lower = data.raw_text.lower()
        palavras_agendamento = ['programado', 'agendado', 'agendamento', 'programada', 'agendada']
        for palavra in palavras_agendamento:
            if palavra in text_lower:
                is_agendado = True
                score -= 0.50
                penalidades.append(f"✗ Comprovante contém '{palavra}' — pagamento apenas agendado, não realizado (-0.50)")
                break

    # Clamp
    score = max(0.0, min(1.0, score))

    # ---- CLASSIFICAÇÃO POR NÍVEL DE NEGÓCIO ----
    # Determina horas desde o comprovante
    horas_diferenca = None
    if data.data_hora:
        dt = parse_data_hora(data.data_hora)
        if dt:
            horas_diferenca = (datetime.now(BRT) - dt).total_seconds() / 3600

    # Verifica se os dados do recebedor conferem
    recebedor_ok = True
    if not data.nome_recebedor or len(data.nome_recebedor) <= 3:
        recebedor_ok = False
    elif not _nomes_correspondem(data.nome_recebedor, EXPECTED_NOME_RECEBEDOR):
        recebedor_ok = False

    cpf_ok = False
    if data.cpf_recebedor:
        cpf_ok, _ = _cpf_fuzzy_match(data.cpf_recebedor, EXPECTED_CPF_RECEBEDOR_PARTIAL)

    dados_basicos_ok = recebedor_ok and data.data_hora and data.id_transacao

    # Verifica se destino confere (nome OU cpf do recebedor bate)
    destino_ok = recebedor_ok or cpf_ok

    if is_agendado:
        nivel = "pixagendado"
    elif score < 0.30:
        nivel = "pixinvalido"
    elif not dados_basicos_ok or score < 0.45:
        nivel = "pixsuspeito"
    elif not destino_ok:
        nivel = "pixdestinoerrado"
    elif not data.valor or data.valor <= 0:
        nivel = "pixsuspeito"
    elif horas_diferenca is not None and horas_diferenca > MAX_HORAS_VALIDADE:
        # Real, acima de 24h (valor não importa)
        nivel = "pixvalidomais24horas"
    elif horas_diferenca is not None and horas_diferenca <= MAX_HORAS_VALIDADE and horas_diferenca > 1:
        # Real, entre 1h e 24h
        if data.valor < 10:
            nivel = "pix24horaspobre"
        elif data.valor > 20:
            nivel = "pix24horaspresente"
        else:
            nivel = "pix24horas"
    elif horas_diferenca is not None and horas_diferenca <= 1:
        # Real, menos de 1h
        if data.valor < 10:
            nivel = "pixpobre"
        elif data.valor > 20:
            nivel = "pixvalidopresente"
        else:
            nivel = "pixvalido"
    else:
        nivel = "pixsuspeito"

    return TrustScore(
        score=round(score, 2),
        nivel=nivel,
        detalhes=detalhes,
        penalidades=penalidades,
    )


def _process_and_validate(file_bytes: bytes, filename: str, endpoint: str) -> ExtractionResult:
    """
    Fluxo unificado: OCR → Parse → Trust Score.
    Usado por todos os endpoints /extract.
    """
    start = time.time()
    logger.info(f"[{endpoint}] Tamanho: {len(file_bytes)} bytes")

    text = extract_text(file_bytes, filename)
    if not text or len(text.strip()) < 10:
        logger.warning(f"[{endpoint}] Texto insuficiente extraído de {filename} — não é comprovante")
        return ExtractionResult(
            success=True,
            dados=PixData(),
            trust=TrustScore(
                score=0.0,
                nivel="naoecomprovante",
                detalhes=["✗ Nenhum texto relevante extraído da imagem"],
                penalidades=["✗ Imagem não contém um comprovante de pagamento"],
            ),
        )

    data = parse_receipt(text)
    trust = calculate_trust_score(data)

    # Registrar ID de transação para detecção de duplicatas futuras
    if data.id_transacao:
        _processed_transaction_ids[data.id_transacao] = time.time()
        # Limpa IDs antigos (>30 dias) para não vazar memória
        cutoff = time.time() - (_DUPLICATE_TTL_HOURS * 3600)
        expired = [k for k, v in _processed_transaction_ids.items() if v < cutoff]
        for k in expired:
            del _processed_transaction_ids[k]

    elapsed = time.time() - start
    logger.info(
        f"[{endpoint}] {filename} | banco={data.banco_origem} | score={trust.score} | nivel={trust.nivel}"
        f" | valor={data.valor} | recebedor={data.nome_recebedor} | pagador={data.nome_pagador}"
        f" | {elapsed:.2f}s"
    )
    if trust.penalidades:
        logger.info(f"[{endpoint}] Penalidades: {'; '.join(trust.penalidades)}")

    return ExtractionResult(success=True, dados=data, trust=trust)


# ============================================================
# ENDPOINTS
# ============================================================

@app.post("/extract", response_model=ExtractionResult)
async def extract_from_upload(file: UploadFile = File(...)):
    """Extrai dados de comprovante PIX enviado como upload."""
    filename = file.filename or "comprovante.jpg"
    logger.info(f"[/extract] Recebido: {filename}")
    try:
        file_bytes = await file.read()
        return _process_and_validate(file_bytes, filename, "/extract")
    except Exception as e:
        logger.error(f"[/extract] Erro ao processar {filename}: {e}")
        return ExtractionResult(success=False, error=str(e))


@app.post("/extract/base64", response_model=ExtractionResult)
async def extract_from_base64(input_data: Base64Input):
    """Extrai dados de comprovante PIX enviado como base64."""
    filename = input_data.filename
    logger.info(f"[/extract/base64] Recebido: {filename}")
    try:
        file_bytes = base64.b64decode(input_data.file)
        return _process_and_validate(file_bytes, filename, "/extract/base64")
    except Exception as e:
        logger.error(f"[/extract/base64] Erro ao processar {filename}: {e}")
        return ExtractionResult(success=False, error=str(e))


@app.post("/extract/url", response_model=ExtractionResult)
async def extract_from_url(input_data: UrlInput):
    """Baixa comprovante de uma URL, analisa e retorna o resultado."""
    start = time.time()
    url = input_data.url
    filename = input_data.filename or url.rsplit('/', 1)[-1].split('?')[0] or "comprovante.jpg"
    logger.info(f"[/extract/url] URL: {url}")
    try:
        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            file_bytes = resp.content
        logger.info(f"[/extract/url] Baixado: {filename} ({len(file_bytes)} bytes)")

        text = extract_text(file_bytes, filename)
        if not text or len(text.strip()) < 10:
            logger.warning(f"[/extract/url] Texto insuficiente de {filename}")
            return ExtractionResult(
                success=False,
                error="N\u00e3o foi poss\u00edvel extrair texto do arquivo baixado.",
            )

        data = parse_receipt(text)
        trust = calculate_trust_score(data)
        elapsed = time.time() - start

        logger.info(
            f"[/extract/url] {filename} | banco={data.banco_origem} | score={trust.score} | nivel={trust.nivel}"
            f" | valor={data.valor} | recebedor={data.nome_recebedor} | pagador={data.nome_pagador}"
            f" | {elapsed:.2f}s"
        )
        if trust.penalidades:
            logger.info(f"[/extract/url] Penalidades: {'; '.join(trust.penalidades)}")

        return ExtractionResult(success=True, dados=data, trust=trust)

    except httpx.HTTPStatusError as e:
        logger.error(f"[/extract/url] Erro HTTP ao baixar {url}: {e.response.status_code}")
        return ExtractionResult(success=False, error=f"Erro ao baixar arquivo: HTTP {e.response.status_code}")
    except httpx.RequestError as e:
        logger.error(f"[/extract/url] Erro de conex\u00e3o ao baixar {url}: {e}")
        return ExtractionResult(success=False, error=f"Erro de conex\u00e3o: {e}")
    except Exception as e:
        logger.error(f"[/extract/url] Erro ao processar {url}: {e}")
        return ExtractionResult(success=False, error=str(e))


@app.post("/ocr", response_model=dict)
async def raw_ocr(file: UploadFile = File(...)):
    """Retorna apenas o texto OCR bruto (debug/desenvolvimento)."""
    filename = file.filename or "file.jpg"
    logger.info(f"[/ocr] Recebido: {filename}")
    try:
        file_bytes = await file.read()
        text = extract_text(file_bytes, filename)
        bank = classify_bank(text)
        logger.info(f"[/ocr] {filename} | banco={bank} | chars={len(text)}")
        return {"text": text, "banco_detectado": bank, "chars": len(text)}
    except Exception as e:
        logger.error(f"[/ocr] Erro ao processar {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    logger.debug("[/health] Health check")
    return {"status": "ok", "version": "1.0.0"}


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
