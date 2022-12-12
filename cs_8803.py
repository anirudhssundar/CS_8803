import torch
from transformers import RobertaTokenizer, RobertaModel
import pdb

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def analyze_1(sent_1, sent_2, sent_3, pos_1, pos_2, pos_3):
    # Analyze polysemy
    sent_1_tokens = tokenizer(sent_1, return_tensors='pt')['input_ids'].to(device)
    sent_2_tokens = tokenizer(sent_2, return_tensors='pt')['input_ids'].to(device)
    sent_3_tokens = tokenizer(sent_3, return_tensors='pt')['input_ids'].to(device)

    out_1 = model(sent_1_tokens).last_hidden_state
    out_2 = model(sent_2_tokens).last_hidden_state
    out_3 = model(sent_3_tokens).last_hidden_state

    head_1 = out_1[0,pos_1,:]
    head_2 = out_2[0,pos_2,:]
    head_3 = out_3[0,pos_3,:]

    head_1 = torch.nn.functional.normalize(head_1.reshape(-1,1), dim=0)
    head_2 = torch.nn.functional.normalize(head_2.reshape(-1,1), dim=0)
    head_3 = torch.nn.functional.normalize(head_3.reshape(-1,1), dim=0)

    print(torch.dot(head_1.squeeze(1), head_2.squeeze(1)))
    print(torch.dot(head_2.squeeze(1), head_3.squeeze(1)))
    print(torch.dot(head_3.squeeze(1), head_1.squeeze(1)))




def analyze_2(sent_4, sent_5, sent_6):
    # Analyze average of last hidden state values
    sent_4_tokens = tokenizer(sent_4, return_tensors='pt')['input_ids'].to(device)
    sent_5_tokens = tokenizer(sent_5, return_tensors='pt')['input_ids'].to(device)
    sent_6_tokens = tokenizer(sent_6, return_tensors='pt')['input_ids'].to(device)

    out_4 = model(sent_4_tokens).last_hidden_state
    out_5 = model(sent_5_tokens).last_hidden_state
    out_6 = model(sent_6_tokens).last_hidden_state

    out_4_mean = out_4.mean(dim=1)
    out_5_mean = out_5.mean(dim=1)
    out_6_mean = out_6.mean(dim=1)

    out_4_norm = torch.nn.functional.normalize(out_4_mean, dim=1)
    out_5_norm = torch.nn.functional.normalize(out_5_mean, dim=1)
    out_6_norm = torch.nn.functional.normalize(out_6_mean, dim=1)

    print(torch.dot(out_4_norm.squeeze(0), out_5_norm.squeeze(0)))
    print(torch.dot(out_4_norm.squeeze(0), out_6_norm.squeeze(0)))
    print(torch.dot(out_6_norm.squeeze(0), out_5_norm.squeeze(0)))  


def analyze_3(sent_4, sent_5, sent_6):
    # Analyze CLS tokens 
    sent_4_tokens = tokenizer(sent_4, return_tensors='pt')['input_ids'].to(device)
    sent_5_tokens = tokenizer(sent_5, return_tensors='pt')['input_ids'].to(device)
    sent_6_tokens = tokenizer(sent_6, return_tensors='pt')['input_ids'].to(device)

    out_4 = model(sent_4_tokens).pooler_output
    out_5 = model(sent_5_tokens).pooler_output
    out_6 = model(sent_6_tokens).pooler_output

    # out_4_mean = out_4.mean(dim=1)
    # out_5_mean = out_5.mean(dim=1)
    # out_6_mean = out_6.mean(dim=1)

    out_4_norm = torch.nn.functional.normalize(out_4, dim=1)
    out_5_norm = torch.nn.functional.normalize(out_5, dim=1)
    out_6_norm = torch.nn.functional.normalize(out_6, dim=1)

    print(torch.dot(out_4_norm.squeeze(0), out_5_norm.squeeze(0)))
    print(torch.dot(out_4_norm.squeeze(0), out_6_norm.squeeze(0)))
    print(torch.dot(out_6_norm.squeeze(0), out_5_norm.squeeze(0)))  

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base').to(device)



sent_1 = 'A woman with a head of cabbage'
sent_2 = 'The head of the woman'
sent_3 = 'A head of cabbage'

analyze_1(sent_1, sent_2, sent_3, pos_1=5, pos_2=2, pos_3=2) # pass in the positiion of word in each sentence
analyze_1(sent_1, sent_2, sent_3, pos_1=5, pos_2=2, pos_3=2) # pass in the positiion of word in each sentence

sent_4 = 'A fighter standing in fire'
sent_5 = 'A firefighter standing in fire'
sent_6 = 'asdkdhbakjd aksjdla alksdjalsk'

analyze_2(sent_4, sent_5, sent_6)
analyze_3(sent_4, sent_5, sent_6)

sent_4 = 'A woman in glasses staring at the sun'
sent_5 = 'A woman in sunglasses'
sent_6 = 'Something random that makes no sense'

analyze_2(sent_4, sent_5, sent_6)
analyze_3(sent_4, sent_5, sent_6)

sent_4 = 'A saber shining in the light'
sent_5 = 'A lightsaber shining'
sent_6 = 'foo bar baz'

analyze_2(sent_4, sent_5, sent_6)
analyze_3(sent_4, sent_5, sent_6)

