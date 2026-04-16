#!/usr/bin/env python3
"""Script simplificado para executar o pipeline"""

import sys
import re
from pathlib import Path

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def extract_pdf_simple(pdf_path):
    """Extrai texto do PDF de forma simples"""
    try:
        import pdfplumber
        print(f"  Extraindo {pdf_path} com pdfplumber...")
        text = ""
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except ImportError:
        pass
    
    try:
        from pypdf import PdfReader
        print(f"  Extraindo {pdf_path} com pypdf...")
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except ImportError:
        print("  ERRO: Instale pdfplumber ou pypdf")
        return ""

def extract_articles_simple(text, lang="pt"):
    """Extrai artigos do texto"""
    articles = {}
    
    # Padrão para encontrar artigos
    pattern = re.compile(r'(?:^|\n)(Art\.\s*(\d+(?:-[A-Z])?)\s*[oº°]?\s)', re.MULTILINE)
    
    matches = list(pattern.finditer(text))
    
    for i, match in enumerate(matches):
        art_num = match.group(2)
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        content = text[start:end].strip()
        
        # Converte para int se for número puro
        if art_num.isdigit():
            art_num = int(art_num)
        
        articles[art_num] = content
    
    return articles

def main():
    print("=" * 60)
    print("Pipeline da Constituição - Português ↔ Nheengatu")
    print("=" * 60)
    
    # Caminhos
    base = Path(__file__).parent
    pt_pdf = base / "data" / "raw" / "constituicao-pt.pdf"
    nhe_pdf = base / "data" / "raw" / "constituicao-nhe.pdf"
    out_dir = base / "output"
    out_dir.mkdir(exist_ok=True)
    
    # Verifica PDFs
    if not pt_pdf.exists():
        print(f"ERRO: PDF Português não encontrado: {pt_pdf}")
        print("Coloque o arquivo em data/raw/constituicao-pt.pdf")
        return 1
    
    if not nhe_pdf.exists():
        print(f"ERRO: PDF Nheengatu não encontrado: {nhe_pdf}")
        print("Coloque o arquivo em data/raw/constituicao-nhe.pdf")
        return 1
    
    print(f"\n✅ PDFs encontrados:")
    print(f"   PT:  {pt_pdf}")
    print(f"   NHE: {nhe_pdf}")
    
    # Extrai texto
    print("\n📄 Extraindo texto...")
    pt_text = extract_pdf_simple(pt_pdf)
    nhe_text = extract_pdf_simple(nhe_pdf)
    
    if not pt_text or not nhe_text:
        print("ERRO: Falha na extração de texto")
        return 1
    
    print(f"   PT:  {len(pt_text):,} caracteres")
    print(f"   NHE: {len(nhe_text):,} caracteres")
    
    # Extrai artigos
    print("\n📚 Extraindo artigos...")
    pt_articles = extract_articles_simple(pt_text, "pt")
    nhe_articles = extract_articles_simple(nhe_text, "nhe")
    
    print(f"   PT:  {len(pt_articles)} artigos")
    print(f"   NHE: {len(nhe_articles)} artigos")
    
    # Alinha artigos
    print("\n🔗 Alinhando artigos...")
    pt_nums = set(pt_articles.keys())
    nhe_nums = set(nhe_articles.keys())
    
    # Converte strings para int quando possível para comparação
    nhe_nums_int = set()
    for n in nhe_nums:
        if isinstance(n, str) and n.isdigit():
            nhe_nums_int.add(int(n))
        else:
            nhe_nums_int.add(n)
    
    common = pt_nums & nhe_nums_int
    print(f"   Artigos comuns: {len(common)}")
    
    # Gera saída
    print("\n💾 Salvando arquivos...")
    
    # TSV com pares de artigos
    tsv_path = out_dir / "article_pairs.tsv"
    with open(tsv_path, 'w', encoding='utf-8') as f:
        f.write("article_num\tpt_text\tnhe_text\n")
        for num in sorted(common, key=lambda x: int(x) if str(x).isdigit() else 0):
            # Encontra a chave correta no dicionário NHE
            nhe_key = num
            if isinstance(num, int) and str(num) in nhe_articles:
                nhe_key = str(num)
            elif num not in nhe_articles:
                # Tenta encontrar como string
                found = False
                for k in nhe_articles:
                    if str(k) == str(num):
                        nhe_key = k
                        found = True
                        break
                if not found:
                    continue
            
            pt_line = pt_articles[num].replace("\n", " ").replace("\t", " ").strip()
            nhe_line = nhe_articles[nhe_key].replace("\n", " ").replace("\t", " ").strip()
            f.write(f"{num}\t{pt_line}\t{nhe_line}\n")
    
    print(f"   ✅ TSV: {tsv_path}")
    
    # JSON
    import json
    json_path = out_dir / "article_pairs.json"
    pairs_list = []
    for num in sorted(common, key=lambda x: int(x) if str(x).isdigit() else 0):
        nhe_key = num
        if isinstance(num, int) and str(num) in nhe_articles:
            nhe_key = str(num)
        elif num not in nhe_articles:
            for k in nhe_articles:
                if str(k) == str(num):
                    nhe_key = k
                    break
            else:
                continue
        pairs_list.append({
            "article": num,
            "pt": pt_articles[num],
            "nhe": nhe_articles[nhe_key]
        })
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(pairs_list, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ JSON: {json_path}")
    
    # Arquivos Moses (um por linha)
    moses_pt = out_dir / "corpus.pt"
    moses_nhe = out_dir / "corpus.nhe"
    with open(moses_pt, 'w', encoding='utf-8') as f_pt, \
         open(moses_nhe, 'w', encoding='utf-8') as f_nhe:
        for num in sorted(common, key=lambda x: int(x) if str(x).isdigit() else 0):
            nhe_key = num
            if isinstance(num, int) and str(num) in nhe_articles:
                nhe_key = str(num)
            elif num not in nhe_articles:
                for k in nhe_articles:
                    if str(k) == str(num):
                        nhe_key = k
                        break
                else:
                    continue
            f_pt.write(pt_articles[num].replace("\n", " ") + "\n")
            f_nhe.write(nhe_articles[nhe_key].replace("\n", " ") + "\n")
    
    print(f"   ✅ Moses: {moses_pt} / {moses_nhe}")
    
    # Estatísticas
    print("\n" + "=" * 60)
    print("RESUMO FINAL")
    print("=" * 60)
    print(f"  Artigos em Português:  {len(pt_articles)}")
    print(f"  Artigos em Nheengatu:  {len(nhe_articles)}")
    print(f"  Pares alinhados:       {len(common)}")
    
    if len(common) > 0:
        print(f"\n  Exemplo (Artigo {list(common)[0]}):")
        first = list(common)[0]
        nhe_key = first
        if isinstance(first, int) and str(first) in nhe_articles:
            nhe_key = str(first)
        elif first not in nhe_articles:
            for k in nhe_articles:
                if str(k) == str(first):
                    nhe_key = k
                    break
        pt_preview = pt_articles[first][:100].replace("\n", " ")
        nhe_preview = nhe_articles[nhe_key][:100].replace("\n", " ")
        print(f"    PT:  {pt_preview}...")
        print(f"    NHE: {nhe_preview}...")
    
    print("\n✅ Pipeline concluído com sucesso!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
