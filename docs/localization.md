# Localization

YOLO Trainer UI supports Traditional Chinese (`zh_TW`, the default) and English
(`en_US`). Change **Language / 語言** on the Settings page; the main interface
switches immediately and the locale is saved in the local `app_settings.json`.

Translations live in [`i18n/en_US.json`](../i18n/en_US.json) and
[`i18n/zh_TW.json`](../i18n/zh_TW.json). Keys are stable semantic identifiers
such as `train.start`; do not use displayed English text as a key.

The lookup order is the selected language, English, then the key itself. This
makes a missing translation safe rather than fatal. Keep CLI arguments, YAML
fields, file paths, raw process logs, and CSV/JSON schema values untranslated.

To add a language, copy `en_US.json`, translate every value without changing
keys or `{placeholder}` names, and register its locale in `I18nManager`.
Validate resources with:

```powershell
.venv\Scripts\python.exe scripts\check_translations.py
```
