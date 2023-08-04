
import sys
import csv
import pandas as pd
import numpy as np
import os
import openai
import time

key_fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/lab_api_key.txt'
with open(key_fname,'r') as file:
    key = file.read()
openai.api_key = key #get it from your openai account


## -------------------------------------------------
# Use class to put together relevant functions
## -------------------------------------------------
class GPT4Writer:
    def __init__(self, model_name="gpt-4-0314", rate_limit_per_minute=3000):
        self.model_name = model_name
        self.rate_limit_per_minute = rate_limit_per_minute
        self.delay = 60.0 / self.rate_limit_per_minute

    def delayed_completion(self, **kwargs):
        """Delay a completion by a specified amount of time."""
        time.sleep(self.delay)
        kwargs.pop('model', None)
        return openai.ChatCompletion.create(model=self.model_name, **kwargs)

    def generate_paragraph(self, prompt):
        """Prompt GPT-4 to generate a single paragraph."""
        content_instruction = "You are a writer that uses a short, concise, and straightforward writing style.  You will be given an incomplete paragraph and you must continue the narrative. To do this, you will immediately continue writing in a way that focuses on conveying specific actions and events that progress the narrative without elaborating on descriptions or emotions. Please write at least 5 sentences in your continuation, focusing on the specific actions and events that are occurring. It is very important that you do not repeat the information already given to you in the prompt. For example, if you are given the sentence, \"I like my coffee with...\" you will answer with \"cream and sugar in the morning. This allows me to be fully alert and ready for the day's work.\""
        completion = self.delayed_completion(
            messages=[
                {"role": "system", "content": content_instruction},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message['content']

    def generate_paragraphs(self, prompt, num_iterations):
        """Generate multiple paragraphs using GPT-4."""
        paras = [self.generate_paragraph(prompt) for _ in range(num_iterations)]
        colnames = [f'gpt4_p{i + 1}' for i in range(num_iterations)]
        df_para = pd.DataFrame(paras, columns=['paras']).T
        df_para.columns = colnames
        df_para.reset_index(drop=True, inplace=True)
        return df_para
    
    def get_paragraph_all(self, df, condition, num_iterations):
        para = df.apply(lambda prompt: self.generate_paragraphs(prompt, num_iterations=num_iterations))
        combined_df = pd.concat([item for item in para], ignore_index=True)
        df_cond = pd.concat([df, combined_df], axis=1)
        df_cond = df_cond.rename(columns={df_cond.columns[0]: 'context'})
        df_cond['condition'] = condition
        condition_col = df_cond.pop('condition')
        df_cond.insert(0, 'condition', condition_col)
        return df_cond



## --------------------------------------------
# read data
## --------------------------------------------
fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/InfoPos/stimuli/InfoPos_Stim_Final_with_Lexical_Characteristics.xlsx'
df = pd.read_excel(fname)
df_sel = df.loc[:183,['info','noinfo','expected','unexpected','LC_pre','HC_uninfo_pre']]
LC_pre = df_sel['LC_pre'].apply(lambda x: x[:-1] if x[-1] == ' ' else x)
LC_info = df_sel['LC_pre'] + ' ' + df_sel['info']
LC_uninfo = df_sel['LC_pre'] + ' ' + df_sel['noinfo']
HC_pre = df_sel['HC_uninfo_pre'].apply(lambda x: x[:-1] if x[-1] == ' ' else x)
HC_exp = df_sel['HC_uninfo_pre'] + ' ' + df_sel['expected']
HC_unexp = df_sel['HC_uninfo_pre'] + ' ' + df_sel['unexpected']



## ---------------------------------------------------
# generate texts for all sentences of all conditions
## ---------------------------------------------------
gpt4_writer = GPT4Writer()
data = [
    (LC_pre, 'LC_pre'),
    (LC_info, 'LC_info'),
    (LC_uninfo, 'LC_uninfo'),
    (HC_pre, 'HC_pre'),
    (HC_exp, 'HC_exp'),
    (HC_unexp, 'HC_unexp')
]
dfs = {}
for dataset, name in data:
    dfs[name] = gpt4_writer.get_paragraph_all(dataset, name, num_iterations=50)
df_gen = pd.concat([dfs[name] for name in ['LC_pre', 'LC_info', 'LC_uninfo', 'HC_pre', 'HC_exp', 'HC_unexp']],axis=0)
data_path=r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/InfoPos/stimuli/'
df_gen.to_csv(data_path+'generated_paragraphs.csv', index=True, index_label='stimID')



'''
## Not in use due to exceeding of request limit
def generate_paragraphs(prompt,num_iterations):
    content_instruction = "You are a writer that uses a short, concise, and straightforward writing style.  You will be given an incomplete paragraph and you must continue the narrative. To do this, you will immediately continue writing in a way that focuses on conveying specific actions and events that progress the narrative without elaborating on descriptions or emotions. Please write at least 5 sentences in your continuation, focusing on the specific actions and events that are occurring. It is very important that you do not repeat the information already given to you in the prompt. For example, if you are given the sentence, \"I like my coffee with...\" you will answer with \"cream and sugar in the morning. This allows me to be fully alert and ready for the day's work.\""
    completion = openai.ChatCompletion.create(
        model="gpt-4-0314",
        n = num_iterations, 
        messages=[
        {"role": "system", "content": content_instruction},
        {"role": "user", "content": prompt}
        ])
    paras = []
    for iter in range(num_iterations):
        paras.append(completion.choices[iter].message.content)
    colnames = [f'gpt4_p{i+1}' for i in range(num_iterations)]
    df_para = pd.DataFrame(paras,columns=['paras']).T
    df_para.columns = colnames
    df_para.reset_index(drop=True,inplace=True)
    return df_para



def generate_paragraphs(prompt, num_iterations):
    # Content instruction to be used for each GPT-4 call
    content_instruction = "You are a writer that uses a short, concise, and straightforward writing style.  You will be given an incomplete paragraph and you must continue the narrative. To do this, you will immediately continue writing in a way that focuses on conveying specific actions and events that progress the narrative without elaborating on descriptions or emotions. Please write at least 5 sentences in your continuation, focusing on the specific actions and events that are occurring. It is very important that you do not repeat the information already given to you in the prompt. For example, if you are given the sentence, \"I like my coffee with...\" you will answer with \"cream and sugar in the morning. This allows me to be fully alert and ready for the day's work.\""

    paras = []
    for iter in range(num_iterations):
        # Make the API call using delayed_completion
        completion = openai.ChatCompletion.create(
            model="gpt-4-0314",
            messages=[
                {"role": "system", "content": content_instruction},
                {"role": "user", "content": prompt}
            ])
        generated_paragraph = completion.choices[0].message['content']
        paras.append(generated_paragraph)

    # Create a DataFrame from the list of paragraphs
    colnames = [f'gpt4_p{i+1}' for i in range(num_iterations)]
    df_para = pd.DataFrame(paras, columns=['paras']).T
    df_para.columns = colnames
    df_para.reset_index(drop=True, inplace=True)

    return df_para


def get_paragraph_all(df, condition, num_iterations):
    para = df.apply(generate_paragraphs, num_iterations = num_iterations)
    combined_df = pd.concat([item for item in para], ignore_index=True)
    df_cond = pd.concat([df,combined_df],axis=1)
    df_cond = df_cond.rename(columns={df_cond.columns[0]: 'context'})
    df_cond['condition'] = condition
    condition_col = df_cond.pop('condition')
    df_cond.insert(0, 'condition', condition_col)
    return df_cond

'''