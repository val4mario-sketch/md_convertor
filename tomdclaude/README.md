# tomdclaude

Sistema di conversione file **modulare** con **validazione gatekeeper**.
Primo convertitore implementato: **Markdown → PDF leggibile** (stile report/articolo).

Affianca il convertitore universale esistente (`tomd2.py` + router) senza
modificarlo: vive nel sottopacchetto `tomdclaude/`.

## Architettura

```
tomdclaude/
├── core/            contratto, risultato, registry, pipeline (gatekeeper)
│   ├── base_converter.py   BaseConverter (ABC): ogni convertitore è una classe
│   ├── result.py           ConversionResult, ValidationIssue
│   ├── registry.py         elenco convertitori + lookup per formato
│   └── pipeline.py         orchestrazione: convert → validate → commit/abort
├── converters/      un file = un convertitore indipendente
│   └── md_to_pdf.py        MarkdownToPdfConverter (markdown → HTML → WeasyPrint)
├── validators/      il "gatekeeper": confronta i metadati
│   ├── md_validator.py     struttura gerarchica dei titoli (MD)
│   ├── pdf_validator.py    integrità del PDF (header, pagine, testo)
│   └── xlsx_validator.py   righe/colonne per foglio (XLSX) — pronto all'uso
├── utils/           logging, runtime GTK, estrazione metadati
├── assets/report.css       foglio di stile del report
├── examples/report_demo.md documento dimostrativo
├── tests/test_md_to_pdf.py primo test case
├── cli.py                  riga di comando
└── logs/                   log dettagliati dei fallimenti
```

## Il gatekeeper (requisito chiave)

Ogni conversione viene scritta su un **file temporaneo**. Diventa definitiva
(commit atomico) **solo** se supera tutti i validatori. Se un validatore rileva
una perdita o incoerenza di dati (es. un titolo perso, un PDF quasi vuoto), il
temporaneo viene **scartato**, il file finale **non** viene salvato e viene
prodotto un **log dettagliato** in `tomdclaude/logs/`.

## MD → PDF: cosa produce

- Copertina con titolo/autore/data (front-matter `title:` / `author:`)
- **Indice automatico** con numeri di pagina (CSS `target-counter`)
- Numerazione automatica dei capitoli (1, 1.1, 1.1.1)
- Intestazione corrente + piè di pagina "Pagina X / Y"
- Tabelle formattate, codice con syntax highlighting (Pygments), riquadri note
- **Matematica LaTeX** `$...$` / `$$...$$` resa in SVG (font Computer Modern,
  resa identica a LaTeX) senza richiedere una toolchain TeX — vedi sotto
- Pulizia automatica dei marcatori di citazione `[cite_start]`/`[cite: N]`
- Tipografia leggibile (serif, testo giustificato, sillabazione)

## Matematica senza LaTeX

Le formule vengono isolate da `pymdownx.arithmatex` (i blocchi di codice non
sono toccati) e rese in **SVG** dal motore *mathtext* di matplotlib
(`mathtext.fontset = cm`). Inline allineate alla linea di base, display centrate.
Una sanitizzazione converte le piccole differenze di sintassi (`\le`→`\leq`,
`\dfrac`→`\frac`, rimozione di `\Big`, `\displaystyle`, ...).

Copertura testata: 236/236 formule sul documento reale di Metodi Numerici.
Le espressioni non renderizzabili non bloccano (di default): restano visibili
come testo grezzo e finiscono nel log; con `--strict-math` diventano bloccanti.

Limite: mathtext non supporta gli ambienti `\begin{matrix|align|cases}` (assenti
nei documenti target). Per quelli servirebbe una toolchain TeX (MiKTeX + dvisvgm).

## Uso

```bash
# dalla cartella che contiene il package ("CONVERTITORE .md")
python -m tomdclaude examples/report_demo.md --to pdf
python -m tomdclaude appunti.md -o build/appunti.pdf --title "Metodi Numerici"
python -m tomdclaude --list

# opzioni utili
#   --no-math          non renderizzare la matematica
#   --strict-math      rifiuta il PDF se una formula non viene resa
#   --keep-citations   non rimuovere i marcatori [cite_start]/[cite: N]

# test (6/6)
python -m tomdclaude.tests.test_md_to_pdf
```

## Dipendenze

`pip install -r tomdclaude/requirements.txt` + runtime GTK3 su Windows
(`winget install --id tschoonj.GTKForWindows -e`). Il modulo `utils/gtk.py`
aggancia GTK automaticamente a runtime.

## Struttura aggiornata (con matematica)

```
converters/math_render.py   MathRenderer: LaTeX -> SVG via matplotlib mathtext
validators/math_validator.py  conta le formule non rese (gatekeeper)
```
