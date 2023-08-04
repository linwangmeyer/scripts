import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bertopic import BERTopic
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer


def create_wordcloud(model, topic):
    text = {word: value for word, value in model.get_topic(topic)}
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def cal_entropy(topics,confidence):
    '''Calculate the entropy of identified topics, weighted by their confidence values'''
    topic_probs = {}
    for topic, prob in zip(topics, confidence):
        if topic in topic_probs:
            topic_probs[topic] += prob
        else:
            topic_probs[topic] = prob
    topic_probs_array = np.array(list(topic_probs.values()))
    normalized_probs = topic_probs_array / np.sum(topic_probs_array)
    entropy = -np.sum(normalized_probs * np.log2(normalized_probs))
    return entropy


## Load pre-trained models
topic_model = BERTopic.load("MaartenGr/BERTopic_Wikipedia")
#topic_model = BERTopic.load("davanstrien/chat_topics")
# for a list of pre-trained topics, see: https://huggingface.co/models?library=bertopic&sort=downloads

    
############################################################
## Load data ##
############################################################
fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/InfoPos/stimuli/InfoTopic_stim_toyData_gpt4_output.csv'
df = pd.read_csv(fname)
df_sel = df.filter(like='gpt4')
    
# ---------------------------------------------------------------------------
# Get the top topic and probability for each paragraph
# Calculate entropy of the topic distribution for all generated paragraphs
# ---------------------------------------------------------------------------
topic_indices = []
probs = []
for index, row in df_sel.iterrows():
    topic_index, prob = topic_model.transform(row)
    topic_indices.append(topic_index)
    probs.append(prob)
result_df = pd.DataFrame({'topic_index': topic_indices, 'prob': probs})
check_df = pd.concat([df['constraint'],result_df],axis=1)

topics_list = check_df['topic_index']
confidence_list = check_df['prob']
entropy_values = []
for topics, confidence in zip(topics_list, confidence_list):
    ev = cal_entropy(topics, confidence)
    entropy_values.append(ev)
check_df['entropy'] = entropy_values

# plot the entropy values for each item
check_df.plot(kind='bar', figsize=(10,8))
plt.xlabel('condition')
plt.ylabel('entropy')
plt.title('topic entropy value by condition')
plt.xticks(check_df.index, check_df['constraint'], rotation=45)
plt.tight_layout()
plt.savefig(r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/InfoPos/plots/Entropy_30.png')
plt.show()

# Use wordcloud to visualize identified topic
create_wordcloud(topic_model, topic=15)

# ---------------------------------------------------------------------------
# Get the top topic and probability for each generated paragraph
# Calculate the topic distributions on a token-level
# ---------------------------------------------------------------------------
row = df_sel.loc[0,'gpt4_p1']
topic_distr, topic_token_distr = topic_model.approximate_distribution(row, use_embedding_model=True, calculate_tokens=True)
df = topic_model.visualize_approximate_distribution(row, topic_token_distr[0])
df

# get continuous topic-document distribution
topic_distr, _ = topic_model.approximate_distribution(row,use_embedding_model=True)
values = topic_distr.flatten()
indices = np.arange(len(values))

# Create a bar plot
plt.bar(indices, values)
arr_flat = topic_distr.flatten()
sorted_indices = np.argsort(arr_flat)[::-1]
sorted_values = arr_flat[sorted_indices]
sorted_indices
for i in np.arange(0,10):
    x = topic_model.get_topic(sorted_indices[i])
    print(x)

topic_distr, _ = topic_model.approximate_distribution(row,min_similarity=0)
topic_model.visualize_distribution(topic_distr[0])

# ------------------------------------------------------
# For the concatenated paragraph: check entropy
# ------------------------------------------------------
df_sel['concatenated'] = df_sel.apply(lambda row: row['gpt4_p1'] + ' ' + row['gpt4_p2'] + ' ' + row['gpt4_p3'] + ' ' + row['gpt4_p4'], axis=1)
topic_distr, _ = topic_model.approximate_distribution(df_sel['concatenated'],use_embedding_model=True)#Nitemx2376
plt.plot(topic_distr[1])

# calculate entropy
def calculate_entropy(probabilities):
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
    return entropy

entropies = np.apply_along_axis(calculate_entropy, axis=1, arr=topic_distr)
df['entropy'] = entropies

# plot the similarity values for each item
plt.figure(figsize=(10,6))
plt.bar(df.index, df['entropy'])
plt.xlabel('condition')
plt.ylabel('entropy')
plt.title('topic entropy value by condition')
plt.xticks(df.index, df['constraint'], rotation=45)
plt.tight_layout()
plt.savefig(r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/InfoPos/plots/Entropy_4.png')
plt.show()
