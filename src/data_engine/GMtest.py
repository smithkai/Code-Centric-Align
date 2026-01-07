import jsonlines

output_file = "data/processed/magicoder_evolved.jsonl"
count = 0
with jsonlines.open(output_file,'r') as reader:
    for item in reader:    
        count += 1

print(count)
