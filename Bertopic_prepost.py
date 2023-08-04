import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bertopic import BERTopic
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
from scipy.stats import ttest_rel

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


def kl_divergence(p, q):
    return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))


def calculate_kl_divergences(topic_distr_pre, topic_distr_post):
    kl_divergences = np.zeros(topic_distr_pre.shape[0])
    for i in range(topic_distr_pre.shape[0]):
        row_pre = topic_distr_pre[i]
        row_post = topic_distr_post[i]
        row_pre /= np.sum(row_pre)
        row_post /= np.sum(row_post)
        kl_divergences[i] = kl_divergence(row_pre, row_post)
    return kl_divergences

def perform_and_print_ttest(data1, data2, label1, label2):
    t_statistic, p_value = ttest_rel(data1, data2)
    print(f"Paired t-test for {label1} vs. {label2}:")
    print(f"t-statistic:{t_statistic:.4f}")
    print(f"p-value:{p_value:.4f}")
    print('--------------')

def get_top_label_and_prob(sorted_indices, topic_distr, item, index):
    top_label = sorted_indices[index]
    top_prob = topic_distr[item, top_label]
    return top_label, top_prob

def calculate_data(topic_distr_pre, topic_distr_post):
    data = []
    index=1
    for item in range(len(topic_distr_pre)):
        sorted_indices_pre = np.argsort(topic_distr_pre[item, :])[::-1]
        sorted_indices_post = np.argsort(topic_distr_post[item, :])[::-1]

        top_label_pre, top_prob_pre = get_top_label_and_prob(sorted_indices_pre, topic_distr_pre, item, index=index)
        top_label_post, top_prob_post = get_top_label_and_prob(sorted_indices_post, topic_distr_post, item, index=index)

        old_label_post, oldlabel_prob_post = get_top_label_and_prob(sorted_indices_pre, topic_distr_post, item, index=index)
        new_label_pre, newlabel_prob_pre = get_top_label_and_prob(sorted_indices_post, topic_distr_pre, item, index=index)

        oldlabel_probdif = oldlabel_prob_post - top_prob_pre
        newlabel_probdif = top_prob_post - newlabel_prob_pre

        item_data = {
            "Top Label Pre": top_label_pre,
            "Top Probability Pre": top_prob_pre,
            "Top Label Post": top_label_post,
            "Top Probability Post": top_prob_post,
            "Old Label Post": old_label_post,
            "Old Probability Post": oldlabel_prob_post,
            "New Label Pre": new_label_pre,
            "New Probability Pre": newlabel_prob_pre,
            "Change of Prob for New Label": newlabel_probdif,
            "Change of Prob for Old Label": oldlabel_probdif
        }

        data.append(item_data)
    df = pd.DataFrame(data)
    return df
# ---------------------------------------------------------------------------
# Load pre-trained models
# ---------------------------------------------------------------------------
topic_model = BERTopic.load("MaartenGr/BERTopic_Wikipedia")
#topic_model = BERTopic.load("davanstrien/chat_topics")
# for a list of pre-trained topics, see: https://huggingface.co/models?library=bertopic&sort=downloads

    
# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/InfoPos/stimuli/InfoPos_Stim_Final_with_Lexical_Characteristics.xlsx'
df = pd.read_excel(fname)
df_sel = df.loc[:183,['info','noinfo','expected','unexpected','LC_pre','HC_uninfo_pre']]
LC_pre = df_sel['LC_pre'].apply(lambda x: x[:-1] if x[-1] == ' ' else x)
LC_info = df_sel['LC_pre'] + ' ' + df_sel['info']
LC_uninfo = df_sel['LC_pre'] + ' ' + df_sel['noinfo']
HC_pre = df_sel['HC_uninfo_pre'].apply(lambda x: x[:-1] if x[-1] == ' ' else x)
HC_exp = df_sel['HC_uninfo_pre'] + ' ' + df_sel['expected']
HC_unexp = df_sel['HC_uninfo_pre'] + ' ' + df_sel['unexpected']


# ---------------------------------------------------------------------------
# Get the top topic distribution for each item
# Calculate the K-L divergence before and after CWs
# ---------------------------------------------------------------------------
# get continuous topic-document distribution
window=20
topic_distr_LCpre, _ = topic_model.approximate_distribution(LC_pre,window=window,use_embedding_model=True)
topic_distr_LCinfo, _ = topic_model.approximate_distribution(LC_info,window=window,use_embedding_model=True)
topic_distr_LCuninfo, _ = topic_model.approximate_distribution(LC_uninfo,window=window,use_embedding_model=True)
topic_distr_HCpre, _ = topic_model.approximate_distribution(HC_pre,window=window,use_embedding_model=True)
topic_distr_HCexp, _ = topic_model.approximate_distribution(HC_exp,window=window,use_embedding_model=True)
topic_distr_HCunexp, _ = topic_model.approximate_distribution(HC_unexp,window=window,use_embedding_model=True)

# Get k-l divergence for pre and post paragraphs
kl_LCpre_LCinfo = calculate_kl_divergences(topic_distr_LCpre, topic_distr_LCinfo)
kl_LCpre_LCuninfo = calculate_kl_divergences(topic_distr_LCpre, topic_distr_LCuninfo)
kl_HCpre_HCexp = calculate_kl_divergences(topic_distr_HCpre, topic_distr_HCexp)
kl_HCpre_HCunexp = calculate_kl_divergences(topic_distr_HCpre, topic_distr_HCunexp)
kl_values = np.vstack((kl_LCpre_LCinfo, kl_LCpre_LCuninfo, kl_HCpre_HCexp, kl_HCpre_HCunexp)).T

# Plot the KL divergence values for each condition
plt.figure(figsize=(10, 6))
plt.boxplot(kl_values, labels=['LCpre vs. LCinfo', 'LCpre vs. LCuninfo', 'HCpre vs. HCexp', 'HCpre vs. HCunexp'])
plt.xlabel('Conditions')
plt.ylabel('KL Divergence')
plt.title('Statistical Comparison of KL Divergence Across Conditions')
plt.tight_layout()
plt.show()

# Get stats
means = kl_values.mean(axis=0)
stds = kl_values.std(axis=0)
conds = ['kl_LCpre_LCinfo','kl_LCpre_LCuninfo','kl_HCpre_HCexp','kl_HCpre_HCunexp']
for i in range(len(means)):
    print(f'{conds[i]}: mean={means[i]:.4f}; std={stds[i]:.4f}')

pairs = [(kl_LCpre_LCinfo, kl_LCpre_LCuninfo, "kl_LCpre_LCinfo", "kl_LCpre_LCuninfo"),
    (kl_LCpre_LCinfo, kl_HCpre_HCexp, "kl_LCpre_LCinfo", "kl_HCpre_HCexp"),
    (kl_LCpre_LCinfo, kl_HCpre_HCunexp, "kl_LCpre_LCinfo", "kl_HCpre_HCunexp"),
    (kl_HCpre_HCexp, kl_HCpre_HCunexp, "kl_HCpre_HCexp", "kl_HCpre_HCunexp")]

for data1, data2, label1, label2 in pairs:
    perform_and_print_ttest(data1, data2, label1, label2)

# ---------------------------------------------------------------------------
# Track the top topics for pre- and post-CW
# ---------------------------------------------------------------------------


# Get data for all conditions
df_LC_pre_uninfo = calculate_data(topic_distr_LCpre, topic_distr_LCuninfo)
df_LC_pre_info = calculate_data(topic_distr_LCpre, topic_distr_LCinfo)
df_HC_pre_exp = calculate_data(topic_distr_HCpre, topic_distr_HCexp)
df_HC_pre_unexp = calculate_data(topic_distr_HCpre, topic_distr_HCunexp)

# Add the 'Condition' column to each data frame
df_LC_pre_info.insert(0, 'Condition', 'LC_pre_info')
df_LC_pre_uninfo.insert(0, 'Condition', 'LC_pre_uninfo')
df_HC_pre_exp.insert(0, 'Condition', 'HC_pre_exp')
df_HC_pre_unexp.insert(0, 'Condition', 'HC_pre_unexp')

stim = {
    'LC_pre': LC_pre,
    'LC_info': LC_info,
    'LC_uninfo': LC_uninfo,
    'HC_pre': HC_pre,
    'HC_exp': HC_exp,
    'HC_unexp': HC_unexp
}
df_stim = pd.DataFrame(stim)

df_out = pd.concat([df_stim,df_LC_pre_info,df_LC_pre_uninfo,df_HC_pre_exp,df_HC_pre_unexp],axis=1)
df_out.insert(0,'itemID',range(1, len(df_out) + 1))
fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/InfoPos/stimuli/InfoPos_BERTopic_trackTopics.csv'
df_out.to_csv(fname, index=False)

# ---------------------------------------------------------------------------
# Get dominant topic and probability values
# Calculate the topic shift on the dominant topics
# ---------------------------------------------------------------------------
# Get the dominant topics
domtopic_LCpre, prob_LCpre = topic_model.transform(LC_pre)
domtopic_LCinfo, prob_LCinfo = topic_model.transform(LC_info)
domtopic_LCuninfo, prob_LCuninfo = topic_model.transform(LC_uninfo)
domtopic_HCpre, prob_HCpre = topic_model.transform(HC_pre)
domtopic_HCexp, prob_HCexp = topic_model.transform(HC_exp)
domtopic_HCunexp, prob_HCunexp = topic_model.transform(HC_unexp)

data = {
    'LC_pre': LC_pre,
    'LC_info': LC_info,
    'LC_uninfo': LC_uninfo,
    'HC_pre': HC_pre,
    'HC_exp': HC_exp,
    'HC_unexp': HC_unexp,
    'kl_LCpre_LCinfo': kl_LCpre_LCinfo,
    'kl_LCpre_LCuninfo': kl_LCpre_LCuninfo,
    'kl_HCpre_HCexp': kl_HCpre_HCexp,
    'kl_HCpre_HCunexp': kl_HCpre_HCunexp,
    'domtopic_LCpre': domtopic_LCpre,
    'prob_LCpre': prob_LCpre,
    'domtopic_LCinfo': domtopic_LCinfo,
    'prob_LCinfo': prob_LCinfo,
    'domtopic_LCuninfo': domtopic_LCuninfo,
    'prob_LCuninfo': prob_LCuninfo,
    'domtopic_HCpre': domtopic_HCpre,
    'prob_HCpre': prob_HCpre,
    'domtopic_HCexp': domtopic_HCexp,
    'prob_HCexp': prob_HCexp,
    'domtopic_HCunexp': domtopic_HCunexp,
    'prob_HCunexp': prob_HCunexp
}

df_out = pd.DataFrame(data)
df_out['itemID'] = range(1, len(df_out) + 1)
column_order = ['itemID'] + [col for col in df_out.columns if col != 'itemID']
df_out = df_out[column_order]
fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/InfoPos/stimuli/InfoPos_BERTopic_measures.csv'
df_out.to_csv(fname, index=False)



