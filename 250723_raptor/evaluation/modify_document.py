import json

pattern = "\n</think>\n\n"
path = "/scratch1/mfp5696/250713_raptor/250723_raptor/raptor_fulldoc_qwen3_8b_modified.json"

# 1) Read & parse the JSON
with open(path, "r") as f:
    data = json.load(f)

# 2) Modify each item
for item in data:
    # only operate if "respond" exists and the pattern is found
    if "respond" in item and pattern in item["respond"]:
        # split off everything before the last occurrence of pattern
        item["respond"] = item["respond"].rsplit(pattern, 1)[1]

# 3) Write the updated JSON back
with open(path, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

