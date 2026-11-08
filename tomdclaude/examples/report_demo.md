title: Technical Report — File Conversion System
author: tomdclaude

# Introduction

This document demonstrates the highly readable **Markdown → PDF** conversion
produced by `tomdclaude`. The goal is not a raw printout of the source but a
*report-style document*: cover, automatic table of contents with page numbers,
chapter numbering, clean tables and code with syntax highlighting.

The text is justified and hyphenated to improve readability on an A4 page.

## Project goals

1. Convert any format to any other, where that makes sense.
2. Keep high fidelity to Markdown for use with an LLM.
3. Never save a file that fails the integrity checks.

## Constraints

- Everything is **free and local**, with no paid APIs.
- No document uploads.
- Works on a machine without a GPU.

# Architecture

The system is organized into independent modules. Each converter is a class
that inherits from `BaseConverter` and exposes a uniform contract.

## Engine comparison

The following table summarizes the technical options evaluated for the
Markdown → PDF path.

| Engine       | Typographic fidelity | CSS control  | Tables | Notes                  |
| :----------- | :------------------- | :----------- | :----- | :--------------------- |
| WeasyPrint   | High                 | Full (CSS)   | Great  | Chosen for reports     |
| LaTeX/Pandoc | Very high            | Indirect     | Good   | Needs a TeX toolchain  |
| ReportLab    | Medium               | Manual       | Manual | Too low-level          |

> WeasyPrint controls the layout with standard CSS, including headers, footers
> and page numbers in the table of contents.

## Code example

The heart of the gatekeeper: the conversion is saved only if it is valid.

```python
def commit_or_abort(tmp, dst, issues):
    if any(i.blocking for i in issues):
        tmp.unlink(missing_ok=True)   # no file if data was lost
        raise ValueError("conversion rejected by the validator")
    os.replace(tmp, dst)              # atomic commit
```

# Validation

The *Validator* module compares the source metadata with that of the product.

!!! note "The gatekeeper principle"
    No file is written to the destination until all the integrity checks have
    passed. Otherwise a detailed error log is produced.

## Implemented checks

- **Heading hierarchy (MD):** the sequence of heading levels must match.
- **PDF integrity:** valid header, at least one page, text in proportion.
- **Rows/columns (XLSX):** no sheet, row or column lost.

# Conclusions

The Markdown → HTML → PDF path with WeasyPrint produces readable, professional
documents, with full control of the style through CSS.
