"""Testa todos os comprovantes da pasta testes/ contra a API."""
import os
import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

API_URL = "http://localhost:8081/extract"
TESTES_DIR = Path(__file__).parent / "testes"

def test_file(filepath):
    """Envia arquivo para API e retorna resultado."""
    import mimetypes
    boundary = "----FormBoundary7MA4YWxkTrZu0gW"
    filename = filepath.name
    mime = mimetypes.guess_type(str(filepath))[0] or "application/octet-stream"
    
    with open(filepath, "rb") as f:
        file_data = f.read()
    
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()
    
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            elapsed = time.time() - start
            result = json.loads(resp.read())
            return result, elapsed
    except Exception as e:
        elapsed = time.time() - start
        return {"success": False, "error": str(e)}, elapsed

def main():
    files = sorted(TESTES_DIR.iterdir())
    files = [f for f in files if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.pdf')]
    
    print(f"{'='*120}")
    print(f"TESTE DE TODOS OS COMPROVANTES ({len(files)} arquivos)")
    print(f"{'='*120}")
    print(f"{'#':<4} {'Arquivo':<55} {'Banco':<12} {'Score':<6} {'Nivel':<28} {'Valor':<10} {'Tempo':<6} {'OK'}")
    print(f"{'-'*120}")
    
    results = []
    for i, f in enumerate(files, 1):
        result, elapsed = test_file(f)
        
        if result.get("success"):
            dados = result["dados"]
            trust = result["trust"]
            banco = (dados.get("banco_origem") or "?")[:11]
            score = trust["score"]
            nivel = trust["nivel"][:27]
            valor = dados.get("valor_raw") or "?"
            n_pen = len(trust.get("penalidades", []))
            ok = "OK" if score >= 0.5 else "LOW"
        else:
            banco = "ERR"
            score = 0
            nivel = result.get("error", "unknown")[:27]
            valor = "?"
            n_pen = 0
            ok = "FAIL"
        
        name = f.name[:54]
        print(f"{i:<4} {name:<55} {banco:<12} {score:<6.2f} {nivel:<28} {valor:<10} {elapsed:<6.1f} {ok}")
        results.append({
            "file": f.name, "banco": banco, "score": score, 
            "nivel": nivel, "valor": valor, "time": round(elapsed, 1), "ok": ok,
            "penalidades": n_pen
        })
    
    print(f"{'='*120}")
    
    # Summary
    total = len(results)
    ok_count = sum(1 for r in results if r["ok"] == "OK")
    low_count = sum(1 for r in results if r["ok"] == "LOW")
    fail_count = sum(1 for r in results if r["ok"] == "FAIL")
    avg_score = sum(r["score"] for r in results) / total if total else 0
    avg_time = sum(r["time"] for r in results) / total if total else 0
    
    print(f"\nRESUMO:")
    print(f"  Total: {total} | OK (>=0.5): {ok_count} | LOW (<0.5): {low_count} | FAIL: {fail_count}")
    print(f"  Score medio: {avg_score:.2f} | Tempo medio: {avg_time:.1f}s")
    
    # Bank distribution
    banks = {}
    for r in results:
        b = r["banco"]
        if b not in banks:
            banks[b] = {"count": 0, "scores": []}
        banks[b]["count"] += 1
        banks[b]["scores"].append(r["score"])
    
    print(f"\n  Por banco:")
    for b, info in sorted(banks.items(), key=lambda x: -x[1]["count"]):
        avg = sum(info["scores"]) / len(info["scores"])
        print(f"    {b:<15} {info['count']:>3} comprovantes | score medio: {avg:.2f}")
    
    # Nivel distribution
    niveis = {}
    for r in results:
        n = r["nivel"]
        niveis[n] = niveis.get(n, 0) + 1
    
    print(f"\n  Por classificacao:")
    for n, count in sorted(niveis.items(), key=lambda x: -x[1]):
        print(f"    {n:<28} {count:>3}")
    
    # Low scores detail
    low_results = [r for r in results if r["score"] < 0.5]
    if low_results:
        print(f"\n  SCORES BAIXOS (<0.5):")
        for r in low_results:
            print(f"    {r['file']:<55} score={r['score']:.2f} nivel={r['nivel']}")

if __name__ == "__main__":
    main()
