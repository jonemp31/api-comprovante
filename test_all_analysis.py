import os, json, subprocess, sys

TESTES_DIR = r"c:\Users\jonat\OneDrive\Documentos\api-comprovante\testes"
URL = "http://127.0.0.1:8000/extract"

files = sorted(os.listdir(TESTES_DIR))
results = []

for i, f in enumerate(files):
    path = os.path.join(TESTES_DIR, f)
    if not os.path.isfile(path):
        continue
    print(f"[{i+1}/{len(files)}] {f}...", end=" ", flush=True)
    try:
        r = subprocess.run(
            ["curl.exe", "-s", "-X", "POST", URL, "-F", f"file=@{path}"],
            capture_output=True, text=True, timeout=60
        )
        data = json.loads(r.stdout)
        success = data.get("success", False)
        trust = data.get("trust", {})
        dados = data.get("dados", {})
        score = trust.get("score", "N/A")
        nivel = trust.get("nivel", "N/A")
        banco = dados.get("banco", "N/A")
        valor = dados.get("valor", "N/A")
        recebedor = dados.get("nome_recebedor", "N/A")
        pagador = dados.get("nome_pagador", "N/A")
        id_tx = dados.get("id_transacao", "N/A")
        chave = dados.get("chave_pix", "N/A")
        data_hora = dados.get("data_hora", "N/A")
        penalidades = trust.get("penalidades", [])
        error = data.get("error", "")
        
        results.append({
            "arquivo": f,
            "success": success,
            "score": score,
            "nivel": nivel,
            "banco": banco,
            "valor": valor,
            "recebedor": recebedor,
            "pagador": pagador,
            "id_transacao": id_tx,
            "chave_pix": chave,
            "data_hora": data_hora,
            "penalidades": penalidades,
            "error": error
        })
        print(f"score={score} nivel={nivel} banco={banco}")
    except Exception as e:
        print(f"ERRO: {e}")
        results.append({"arquivo": f, "success": False, "error": str(e), "score": "ERR", "nivel": "ERR"})

# Save full results
with open("analysis_results.json", "w", encoding="utf-8") as fp:
    json.dump(results, fp, ensure_ascii=False, indent=2)

# Print summary table
print("\n" + "="*120)
print(f"{'ARQUIVO':<55} {'SCORE':>6} {'NIVEL':<25} {'BANCO':<15} {'VALOR':>10}")
print("="*120)
for r in results:
    f = r["arquivo"][:54]
    s = str(r.get("score","?"))[:6]
    n = str(r.get("nivel","?"))[:24]
    b = str(r.get("banco","?"))[:14]
    v = str(r.get("valor","?"))[:10]
    print(f"{f:<55} {s:>6} {n:<25} {b:<15} {v:>10}")

print(f"\nTotal: {len(results)} arquivos")
print(f"Resultados completos salvos em analysis_results.json")
