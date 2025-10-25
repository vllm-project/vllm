When passing JSON CLI arguments, the following sets of arguments are equivalent:

- `--json-arg '{"key1": "value1", "key2": {"key3": "value2"}}'`
- `--json-arg.key1 value1 --json-arg.key2.key3 value2`

Additionally, list elements can be passed individually using `+`:

- `--json-arg '{"key4": ["value3", "value4", "value5"]}'`
- `--json-arg.key4+ value3 --json-arg.key4+='value4,value5'`