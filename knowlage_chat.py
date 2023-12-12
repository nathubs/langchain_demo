import pickle
import erniebot

from tqdm import tqdm
from typing import List
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.faiss import FAISS

erniebot.api_type = 'aistudio'
erniebot.access_token = '19d47a3bf99d0c29a3437b83e567e5bef74659c2'

class ErniebotEmbeddings(Embeddings):
    def __init__(self, access_token: str, api_type: str = 'aistudio', model: str = 'ernie-text-embedding', batch_size: int = 16, embedding_size: int = 384):
        erniebot.api_type = api_type
        erniebot.access_token = access_token
        self.model = model
        self.batch_size = batch_size
        self.embedding_size = embedding_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_texts = [texts[i: i+self.batch_size]
                       for i in range(0, len(texts), self.batch_size)]
        embeddings = []
        for batch_text in tqdm(batch_texts):
            try:
                response = erniebot.Embedding.create(
                    model=self.model,
                    input=batch_text
                )
                embeddings += [item['embedding'] for item in response['data']]
            except:
                for text in tqdm(batch_text):
                    try:
                        response = erniebot.Embedding.create(
                            model=self.model,
                            input=[text]
                        )
                        embeddings.append(response['data'][0]['embedding'])
                    except:
                        embeddings.append([0.0] * self.embedding_size)
                        continue
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = erniebot.Embedding.create(
            model=self.model,
            input=[text]
        )
        return response['data'][0]['embedding']


embedding = ErniebotEmbeddings(erniebot.access_token)

data_dir = 'index'

index = FAISS.load_local(data_dir, embedding)

query = input('> ')

k = 10
search_type = 'similarity' # mmr

prompt = '请根据如下问题和参考的飞桨 (PaddlePaddle) 文档给出回答。\n问题：%s\n参考：\n' % query

references = []
for i, doc in enumerate(index.search(query, k=k, search_type=search_type)):
    prompt += '[%d] %s\n%s\n\n' % (i+1, doc.metadata['source'], doc.page_content)
    reference = doc.metadata['source'].replace('docs-develop', 'https://github.com/PaddlePaddle/docs/blob/develop')
    if reference not in references:
        references.append(reference)

prompt += '\n回答：\n'
response = erniebot.ChatCompletion.create(
    model='ernie-bot-4',
    messages=[{
                'role': 'user',
                'content': prompt
    }],
    stream=True
)

for chunk in response:
    print(chunk['result'], end='')

print('\n\n\n## 参考文档\n' + '\n'.join(['%d. [%s](%s)' % (i+1, item.strip(), item.replace(" ", '%20')) for i, item in enumerate(references)]))