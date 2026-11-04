title: Relazione Tecnica — Sistema di Conversione File
author: tomdclaude

# Introduzione

Questo documento dimostra la conversione **Markdown → PDF** ad alta leggibilità
prodotta da `tomdclaude`. L'obiettivo non è una stampa grezza del sorgente, ma un
*documento in stile report*: copertina, indice automatico con numeri di pagina,
numerazione dei capitoli, tabelle pulite e codice con evidenziazione della sintassi.

Il testo è giustificato e sillabato per migliorare la leggibilità su pagina A4.

## Obiettivi del progetto

1. Convertire qualunque formato in qualunque altro, dove ha senso.
2. Mantenere l'alta fedeltà verso Markdown per l'uso con un LLM.
3. Non salvare mai un file che non superi i controlli di integrità.

## Vincoli

- Tutto **gratuito e in locale**, nessuna API a pagamento.
- Nessun upload dei documenti.
- Funzionamento su macchina senza GPU.

# Architettura

Il sistema è organizzato in moduli indipendenti. Ogni convertitore è una classe
che eredita da `BaseConverter` ed espone un contratto uniforme.

## Confronto fra i motori

La tabella seguente riassume le scelte tecniche valutate per il percorso
Markdown → PDF.

| Motore       | Fedeltà tipografica | Controllo CSS | Tabelle | Note                      |
| :----------- | :------------------ | :------------ | :------ | :------------------------ |
| WeasyPrint   | Alta                | Totale (CSS)  | Ottime  | Scelto per i report       |
| LaTeX/Pandoc | Altissima           | Indiretto     | Buone   | Richiede una toolchain TeX |
| ReportLab    | Media               | Manuale       | Manuali | Troppo basso livello      |

> WeasyPrint permette di controllare l'impaginazione con CSS standard, incluse
> intestazioni, piè di pagina e numeri di pagina nell'indice.

## Esempio di codice

Il cuore del gatekeeper: la conversione viene salvata solo se valida.

```python
def commit_or_abort(tmp, dst, issues):
    if any(i.blocking for i in issues):
        tmp.unlink(missing_ok=True)   # niente file in caso di perdita dati
        raise ValueError("conversione rifiutata dal validator")
    os.replace(tmp, dst)              # commit atomico
```

# Validazione

Il modulo *Validator* confronta i metadati del sorgente con quelli del prodotto.

!!! note "Principio del gatekeeper"
    Nessun file viene scritto sulla destinazione finché tutti i controlli di
    integrità non sono superati. In caso contrario viene prodotto un log
    dettagliato dell'errore.

## Controlli implementati

- **Struttura gerarchica (MD):** la sequenza dei livelli dei titoli deve coincidere.
- **Integrità PDF:** header valido, almeno una pagina, testo non sproporzionato.
- **Righe/colonne (XLSX):** nessun foglio, riga o colonna persa.

# Conclusioni

Il percorso Markdown → HTML → PDF con WeasyPrint produce documenti leggibili e
professionali, mantenendo il pieno controllo dello stile tramite CSS.
