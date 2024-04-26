from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import numpy as np
from tqdm import tqdm

model_name = "/disk2/ysun/llm/finetune/Mistral-7b-s" # PATH TO LOAD A MODEL
outputfile = 'test_prompt_data_hard_mistral_finetune.csv' # PATH TO SAVE THE OUTPUT FILE

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load base model
based_model = AutoModelForCausalLM.from_pretrained(model_name, device_map= "auto")
# load test data
test_prompt_data = pd.read_csv("./data/sensitivity/test_prompt_data.csv")
fewshot_set = test_prompt_data.iloc[219:,:] # DEFINE A FEWSHOT SET

def get_few_shot(fewshot_set, num=5):
    fewshot = ''
    idx_list = []
    for i in range(num):
        idx = np.random.randint(0,len(fewshot_set))
        answer = fewshot_set.iloc[idx].answer
        question = fewshot_set.iloc[idx].prompt.split("[Reasoning].")[1].replace('?',answer)#.replace("\n", "")
        fewshot += f'\n\nExample {i+1}:\n'
        fewshot += question
        idx_list.append(idx)
    
    return fewshot.strip(), idx_list


def get_few_shot_cell_sim(input_prompt, fewshot_set, num=5):
    fewshot = ''
    input_mutation = input_prompt.split("The mutations of the cell line are ")[1].split(".\nDrug Sensitivity")[0].split(", ")
    similarity_score_list = []
    for i in range(len(fewshot_set)):
        temp_prompt = fewshot_set.iloc[i].prompt
        temp_mutation = temp_prompt.split("The mutations of the cell line are ")[1].split(".\nDrug Sensitivity")[0].split(", ")
        intersection = set(input_mutation).intersection(set(temp_mutation))
        sim = (len(intersection) / len(set(input_mutation))) * 100 / len(set(temp_mutation))
        similarity_score_list.append(sim)
    sorted_indices = sorted(enumerate(similarity_score_list), key=lambda x: x[1], reverse=True)
    top_idx_lst = [index for index, _ in sorted_indices[:num]]
    for idx in top_idx_lst:
        answer = fewshot_set.iloc[idx].answer
        question = fewshot_set.iloc[idx].prompt.split("[Reasoning].")[1].replace('?',answer)#.replace("\n", "")
        fewshot += f'\n\nExample {idx+1}:\n'
        fewshot += question

    return fewshot.strip(), top_idx_lst


def evaluate_test_data(idx):
    pipe = pipeline(task="text-generation", model=based_model, tokenizer=tokenizer, max_length=500)

    instruction = "Think step by step and decide in a single word reflecting the drug sensitivity of the drug on the cell line: [Sensitive/Resistant]"
    prompt = test_prompt_data.prompt[idx]       
    prompt_content = f"<s>[INST] <<SYS>>{instruction}<</SYS>>{prompt}[/INST]"

    # Run prompt and pipeline
    result = pipe(prompt_content)
    output = result[0]['generated_text'][len(prompt_content):]
    answer = test_prompt_data.answer[idx]
    return test_prompt_data.cell_id[idx], prompt_content, answer, output


def evaluate_test_data_fewshot(idx, num_fewshot=5, sample_type = 'random'):
    pipe = pipeline(task="text-generation", model=based_model, tokenizer=tokenizer, max_length=5000)

    instruction = "Think step by step and decide in a single word reflecting the drug sensitivity of the drug on the cell line: [Sensitive/Resistant]"
    prompt = test_prompt_data.prompt[idx]       
    if sample_type == 'random':
        fewshot, idx_list = get_few_shot(fewshot_set, num_fewshot)
    elif sample_type == 'similar_mutation':
        fewshot, idx_list = get_few_shot_cell_sim(prompt, fewshot_set, num_fewshot)

    prompt_content = f"<s>[INST] <<SYS>>{instruction}\n{fewshot}<</SYS>>{prompt}[/INST]"

    # Run prompt and pipeline
    result = pipe(prompt_content)
    output = result[0]['generated_text'][len(prompt_content):]
    answer = test_prompt_data.answer[idx]
    return test_prompt_data.cell_id[idx], prompt_content, answer, output, idx_list

def get_few_shot_by_idx(fewshot_set, idx_list):
    instruction = "Think step by step and decide in a single word reflecting the drug sensitivity of the drug on the cell line: [Sensitive/Resistant]"
    fewshot = ''
    for i, idx in enumerate(idx_list):
        answer = fewshot_set.iloc[idx].answer
        question = fewshot_set.iloc[idx].prompt.split("[Reasoning].")[1]#.replace('?',answer)#.replace("\n", "")
        fewshot += f"<s>[INST] <<SYS>>{instruction}<</SYS>>{question}[/INST]{answer}</s>" #f'\n\nExample {i+1}:\n'
        # fewshot += question#.strip()
    
    return fewshot.strip(), idx_list


def get_few_shot_cell_sim2(input_prompt, fewshot_set, num=5):
    instruction = "Think step by step and decide in a single word reflecting the drug sensitivity of the drug on the cell line: [Sensitive/Resistant]"
    fewshot = ''
    input_mutation = input_prompt.split("The mutations of the cell line are ")[1].split(".\nDrug Sensitivity")[0].split(", ")
    similarity_score_list = []
    for i in range(len(fewshot_set)):
        temp_prompt = fewshot_set.iloc[i].prompt
        temp_mutation = temp_prompt.split("The mutations of the cell line are ")[1].split(".\nDrug Sensitivity")[0].split(", ")
        intersection = set(input_mutation).intersection(set(temp_mutation))
        sim = (len(intersection) / len(set(input_mutation))) * 100 / len(set(temp_mutation))
        similarity_score_list.append(sim)
    sorted_indices = sorted(enumerate(similarity_score_list), key=lambda x: x[1], reverse=True)
    top_idx_lst = [index for index, _ in sorted_indices[:num]]
    for idx in top_idx_lst:
        answer = fewshot_set.iloc[idx].answer
        question = fewshot_set.iloc[idx].prompt.split("[Reasoning].")[1]#.replace('?',answer)#.replace("\n", "")
        fewshot += f"<s>[INST] <<SYS>>{instruction}<</SYS>>{question}[/INST]{answer}</s>"#f'\n\nExample {idx+1}:\n'
        # fewshot += question

    return fewshot.strip(), top_idx_lst


def get_few_shot_drug_cell_sim(input_prompt, fewshot_set, num=5):
    input_drug = input_prompt.split("The drug is ")[1].split(". The drug SMILES")[0]
    fewshot_set.loc[:,"drug"] = fewshot_set.prompt.apply(lambda x: x.split("The drug is ")[1].split(". The drug SMILES")[0])
    fewshot_set_same_drug = fewshot_set.loc[fewshot_set.drug == input_drug].copy()
    fewshot_set_diff_drug = fewshot_set.loc[fewshot_set.drug != input_drug].copy()
    if len(fewshot_set_same_drug) >= num:
        fewshot, idx_lst = get_few_shot_cell_sim2(input_prompt, fewshot_set_same_drug, num)
    else:
        avail_num = len(fewshot_set_same_drug)
        fewshot, idx_lst = get_few_shot_cell_sim2(input_prompt, fewshot_set_same_drug, avail_num)
        fewshot2, idx_lst2 = get_few_shot_cell_sim2(input_prompt, fewshot_set_diff_drug, num-avail_num)
        fewshot += fewshot2.strip()
        idx_lst += idx_lst2
    return fewshot.strip(), idx_lst


def evaluate_test_data_fewshot_2(idx, num_fewshot=5, sample_type = 'random'):
    pipe = pipeline(task="text-generation", model=based_model, tokenizer=tokenizer, max_length=5000)

    instruction = "Think step by step and decide in a single word reflecting the drug sensitivity of the drug on the cell line: [Sensitive/Resistant]"
    prompt = test_prompt_data.prompt[idx].split("[Reasoning].")[1].replace("\n", "")       
    
    if type(sample_type) == list:
        fewshot, idx_list =  get_few_shot_by_idx(fewshot_set, sample_type)
        # idx_list = sample_type
    elif sample_type == 'random':
        fewshot, idx_list = get_few_shot(fewshot_set, num_fewshot)
    elif sample_type == 'similar_mutation':
        fewshot, idx_list = get_few_shot_cell_sim2(prompt, fewshot_set, num_fewshot)
    else:
        fewshot, idx_list = get_few_shot_drug_cell_sim(prompt, fewshot_set, num_fewshot)

    prompt_content = fewshot+f"<s>[INST] <<SYS>>{instruction}<</SYS>>{prompt}[/INST]" 

    # Run prompt and pipeline
    result = pipe(prompt_content, temperature=0.0)
    output = result[0]['generated_text'][len(prompt_content):].strip()
    answer = test_prompt_data.answer[idx]
    return test_prompt_data.cell_id[idx], prompt_content, answer, output, idx_list


def inference_zeroshot():
    N = 200
    cell_list = []
    prompt_list = []
    output_list = []
    answer_list = []
    for i in tqdm(range(N)):
        c, p, a, o = evaluate_test_data(i)
        print(c, p, a, o)
        cell_list.append(c)
        prompt_list.append(p)
        answer_list.append(a)
        output_list.append(o)

    df = pd.DataFrame({'cell_id': cell_list, 'prompt': prompt_list,  'answer': answer_list, 'output': output_list})
    df.to_csv(f'./data/sensitivity/{outputfile}', index=None)


def inference_fewshot(num_fewshot=15, sample_type = 'random'):
    N = 200
    cell_list = []
    prompt_list = []
    output_list = []
    answer_list = []
    index_list = []
    for i in tqdm(range(N)):
        c, p, a, o, idx = evaluate_test_data_fewshot(i, num_fewshot, sample_type)
        print(c, p, a, o)
        cell_list.append(c)
        prompt_list.append(p)
        answer_list.append(a)
        output_list.append(o)
        index_list.append(idx)

    df = pd.DataFrame({'cell_id': cell_list, 'prompt': prompt_list,  'answer': answer_list, 'output': output_list, 'fewshot_idx': index_list})
    df.to_csv(f'./data/sensitivity/{outputfile.format(num_fewshot=num_fewshot)}', index=None)


def inference_fewshot2(num_fewshot=15, sample_type = 'random'):
    N = 200
    cell_list = []
    prompt_list = []
    output_list = []
    answer_list = []
    index_list = []
    for i in tqdm(range(N)):
        c, p, a, o, idx = evaluate_test_data_fewshot_2(i, num_fewshot, sample_type)
        print(c, p, a, o)
        cell_list.append(c)
        prompt_list.append(p)
        answer_list.append(a)
        output_list.append(o)
        index_list.append(idx)

    df = pd.DataFrame({'cell_id': cell_list, 'prompt': prompt_list,  'answer': answer_list, 'output': output_list, 'fewshot_idx': index_list})
    df.to_csv(f'./data/sensitivity/{outputfile.format(num_fewshot=num_fewshot)}', index=None)


if __name__ == '__main__':
    inference_zeroshot()
    inference_fewshot2(5, 'similar_mutation2')
    

