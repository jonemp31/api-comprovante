import requests, os

pasta = r'c:\Users\jonat\OneDrive\Documentos\api-comprovante\testes'
arquivos = sorted(os.listdir(pasta))

print("ARQUIVO|BANCO|SCORE|NIVEL|CAMPOS|PENDENCIAS")
print("-" * 120)

for f in arquivos:
    path = os.path.join(pasta, f)
    try:
        with open(path, 'rb') as fh:
            r = requests.post('http://127.0.0.1:8000/extract', files={'file': (f, fh)}, timeout=120)
        d = r.json()
        dados = d.get('dados', {})
        trust = d.get('trust', {})

        campos = []
        if dados.get('nome_recebedor'): campos.append('NomeRec')
        if dados.get('cpf_recebedor'): campos.append('CPFRec')
        if dados.get('instituicao_recebedor'): campos.append('InstRec')
        if dados.get('nome_pagador'): campos.append('NomePag')
        if dados.get('cpf_pagador'): campos.append('CPFPag')
        if dados.get('instituicao_pagador'): campos.append('InstPag')
        if dados.get('valor'): campos.append('Valor')
        if dados.get('data_hora'): campos.append('Data')
        if dados.get('id_transacao'): campos.append('ID')
        if dados.get('chave_pix'): campos.append('Chave')

        pends = []
        for p in trust.get('penalidades', []):
            if 'Nome do recebedor' in p or 'Nome recebedor' in p:
                pends.append('!NomeRec')
            elif 'Nome do pagador' in p or 'Nome pagador' in p:
                pends.append('!NomePag')
            elif 'CPF do recebedor' in p:
                pends.append('!CPFRec')
            elif 'NÃO contém trecho' in p:
                pends.append('!CPFErrado')
            elif 'CPF do pagador' in p:
                pends.append('!CPFPag')
            elif 'Valor' in p and 'encontrado' in p:
                pends.append('!Valor')
            elif 'Data' in p and 'encontrada' in p:
                pends.append('!Data')
            elif 'mais de 24h' in p:
                pends.append('>24h')
            elif 'ID de trans' in p:
                pends.append('!ID')
            elif 'Banco emissor' in p:
                pends.append('!Banco')
            elif 'Instituição do recebedor' in p:
                pends.append('!InstRec')
            elif 'Chave PIX' in p and 'encontrada' in p:
                pends.append('!Chave')
            elif 'menciona' in p:
                pends.append('!TipoPIX')
            elif 'agendad' in p.lower() or 'programad' in p.lower():
                pends.append('AGENDADO')

        banco = dados.get('banco_origem', '?')
        score = trust.get('score', 0)
        nivel = trust.get('nivel', '?')
        campos_str = ', '.join(campos) if campos else '-'
        pends_str = ', '.join(pends) if pends else '-'
        
        print(f"{f}|{banco}|{score}|{nivel}|{campos_str}|{pends_str}")
    except Exception as e:
        print(f"{f}|ERRO|0|erro|-|{str(e)[:80]}")
