# Documentation Translations

This directory stores Sphinx/gettext translation catalogs. English `.rst` files
under `docs/source/` remain the source of truth.

## Directory Layout

- `zh_CN/`: Simplified Chinese translations.
- `LC_MESSAGES/`: Standard gettext catalog directory.
- `.po`: Reviewable translation files generated/updated by `sphinx-intl`.

## Translation Flow

1. `sphinx-build -b gettext` extracts English strings into `.pot` templates (throwaway).
2. `sphinx-intl update` writes new strings as empty entries and changed strings as `fuzzy` in the committed `.po` files.
3. `tools/translate_docs_zh.py` calls the model to fill in the empty and fuzzy entries.
4. A PR is opened on `automation/update-chinese-docs` for human review.
5. After merge, `build_doc.yml` rebuilds the site at `/zh_CN/`.

Steps 1–4 run weekly in `translate_doc_zh.yml` (Monday 09:00 UTC, or manual
dispatch). Step 5 runs automatically on push to `dev`.

To change the translation prompt, edit the system message in
`tools/translate_docs_zh.py`.

This README is for maintainers only. It is not linked from the Sphinx toctree,
so it should not appear on the public documentation website.
