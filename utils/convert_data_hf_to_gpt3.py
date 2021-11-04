import json

def convert_from_hf_to_gpt3(input_folder, output_folder):
    for split in ['val', 'test','train']:
        print(f"working on {split}...")
        count = 0
        with open(output_folder+"/"+split+".json", 'w') as f_out:
            with open(input_folder+"/"+split+".source") as f1, open(input_folder+"/"+split+".target") as f2:
                for src_text,tgt_text in zip(f1, f2):
                    src_text = src_text.rstrip()
                    tgt_text = tgt_text.rstrip()
                    data = {"context":src_text+"\n\n", "completion": tgt_text+"<|endoftext|>"}
                    f_out.write(json.dumps(data)+'\n')
                    count+=1
        print(f"{count} lines in total")


convert_from_hf_to_gpt3("train_data", "train_data_gpt3")