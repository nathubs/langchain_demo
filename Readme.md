# 介绍
对于大模型而言，它所拥有的知识是来源于它的训练数据，也就是说目前大模型还没有不断学习新知识的能力

那么，如果想要引入一些未曾经过训练的数据，一般有几种做法，一是微调训练，二是使用知识库

本次就介绍一下如何结合 embedding API 和 langchain 构建一个大模型知识库

结合飞桨的官方文档，实现一个飞桨 PaddlePaddle 文档问答助手

# 快速体验
已将本项目制作成 Gradio 应用并部署于 AIStudio 平台上

应用链接：飞桨PaddlePaddle文档问答助手

LangChain
LangChain 是一个强大的框架，旨在帮助开发人员使用语言模型构建端到端的应用程序。

它提供了一套工具、组件和接口，可简化创建由大型语言模型 (LLM) 和聊天模型提供支持的应用程序的过程。

LangChain 的主要特点包括：

数据感知：将语言模型连接到其他数据源

具有代理性：允许语言模型与其环境进行交互

组件：LangChain 提供了与语言模型一起工作所需的组件的模块化抽象

针对特定用例的链：链可以被认为是以特定方式组装这些组件，以便最好地完成特定用例

LangChain 的使用场景主要与语言模型的一般使用场景重叠，包括文档分析和摘要，聊天机器人和代码分析。

无论您是否使用 LangChain 框架的其余部分，这些组件都设计得易于使用。这些链也设计为可定制。

总的来说，LangChain 是一个非常强大且灵活的框架，可以帮助开发人员更轻松地利用大型语言模型（LLM）来构建各种应用程序。

如果你对更多细节感兴趣，你可以查看他们的官方文档。

环境配置
解压文档
文档来源：PaddlePaddle/docs
In [19]
!unzip -q docs-develop.zip 
安装依赖
LangChain：构建知识库

Erniebot：文心一言 SDK

Faiss：向量匹配库

```
!pip install faiss-cpu -q
!pip install langchain -q
!pip install erniebot -q
```
参数配置
配置文心一言 SDK 的个人 Access Token

个人 Access Token 可在 AIStudio 个人中心 获取

```
import erniebot

erniebot.api_type = 'aistudio'
erniebot.access_token = ''
```
文档对话助手
构建文本向量库
使用 LangChain 读取并切分文档为文本片段

使用文心一言 SDK 对所有文本片段进行向量编码

构建一个简单的文本向量库并保存

PS. 完整完成所有文档片段编码需要大约 40 分钟

```
import os
import time
import pickle
import erniebot

from tqdm import tqdm

from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


docs_dir = 'docs-develop'
data_dir = 'index'

assert not os.path.exists(os.path.join(data_dir, 'data.pkl')), '如需重新编码，请删除这段代码'

exts = ['md', 'rst']
batch_size = 16
chunk_size = 384
loader_cls = TextLoader

docs = []
embeddings = []
texts = []
metadatas = []

for ext in exts:
    loader = DirectoryLoader(
        docs_dir,
        glob='*.%s' % ext,
        recursive=True,
        show_progress=True,
        silent_errors=True,
        loader_cls=loader_cls
    )
    docs += loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
docs = splitter.split_documents(docs)

batch_docs = [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]

for batch_doc in tqdm(batch_docs):
    try:
        response = erniebot.Embedding.create(
            model='ernie-text-embedding',
            input=[item.page_content for item in batch_doc]
        )
        embeddings += [item['embedding'] for item in response['data']]
        texts += [item.page_content for item in batch_doc]
        metadatas += [item.metadata for item in batch_doc]
        time.sleep(1)
    except:
        for text in tqdm(batch_doc):
            try:
                response = erniebot.Embedding.create(
                    model='ernie-text-embedding',
                    input=[text.page_content]
                )
                embeddings.append(response['data'][0]['embedding'])
                texts.append(text.page_content)
                metadatas.append(text.metadata)
                time.sleep(1)
            except:
                continue

data = {
    "embeddings": embeddings,
    "texts": texts,
    'metadatas': metadatas,
}

with open(os.path.join(data_dir, 'data.pkl'), 'wb') as f:
    pickle.dump(data, f)
```
构建知识库
使用 LangChain 和 Faiss 构建一个向量索引知识库

```
import os
import pickle
import erniebot

from tqdm import tqdm
from typing import List
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.faiss import FAISS


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


data_dir = 'index'
embedding = ErniebotEmbeddings(erniebot.access_token)


with open(os.path.join(data_dir, 'data.pkl'), 'rb') as f:
    data = pickle.load(f)

index = FAISS._FAISS__from(
    texts=data['texts'], embeddings=data['embeddings'], embedding=embedding, metadatas=data['metadatas']
)

index.save_local(data_dir)
```
文档问答
结合向量检索知识库和文心一言对话 API

通过匹配与提问相关的文档片段，结合执行文档片段作为大模型输入，再获取大模型的回答

回答最后还将匹配到的文档列表一并输出，供用户参考

```
import pickle
import erniebot

from tqdm import tqdm
from typing import List
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.faiss import FAISS


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
    model='ernie-bot',
    messages=[{
                'role': 'user',
                'content': prompt
    }],
    stream=True
)

for chunk in response:
    print(chunk['result'], end='')

print('\n\n\n## 参考文档\n' + '\n'.join(['%d. [%s](%s)' % (i+1, item.strip(), item.replace(" ", '%20')) for i, item in enumerate(references)]))
```
>  如何安装 Paddle
根据您提供的资源和参考文档，以下是如何安装Paddle的回答：

要安装Paddle，您可以按照以下步骤进行操作：

1. 确定您的硬件平台：根据您的计算机是否具有NVIDIA® GPU，您可以选择安装CPU版本或GPU版本的PaddlePaddle。如果您的计算机没有NVIDIA® GPU，您可以安装CPU版本的PaddlePaddle。如果有NVIDIA® GPU，请确保满足相关的编译条件。
2. 卸载旧版本：如果您之前已经安装了PaddlePaddle，请先卸载旧版本。对于CPU版本，您可以使用以下命令卸载：


```bash
pip uninstall paddlepaddle
```
对于GPU版本，您可以使用以下命令卸载：


```bash
pip uninstall paddlepaddle-gpu
```
如果您同时安装了CPU和GPU版本，请根据实际情况选择相应的卸载命令。
3. 安装依赖项：在安装PaddlePaddle之前，您需要先安装一些依赖项。这些依赖项包括NumPy、Cython等。您可以通过以下命令安装这些依赖项：


```bash
pip install numpy cython
```
4. 安装PaddlePaddle：根据您的硬件平台和需求，选择相应的安装命令。对于CPU版本，您可以使用以下命令安装：


```bash
pip install paddlepaddle
```
对于GPU版本，您可以使用以下命令安装：


```bash
pip install paddlepaddle-gpu
```
如果您希望同时安装CPU和GPU版本，请使用以下命令：


```bash
pip install paddlepaddle paddlepaddle-gpu
```
5. 验证安装：安装完成后，您可以验证PaddlePaddle是否正确安装。您可以通过运行以下命令来测试PaddlePaddle的基本功能：


```python
import paddle
print(paddle.__version__)
```
如果输出了PaddlePaddle的版本号，则表示安装成功。

以上是安装PaddlePaddle的一般步骤。请注意，在某些情况下，可能还需要根据您的操作系统和环境进行额外的配置或调整。有关更详细的说明和指南，请参考PaddlePaddle官方文档中的相关章节。


## 参考文档
1. [https://github.com/PaddlePaddle/docs/blob/develop/docs/eval/【Hackathon No.113】 PR.md](https://github.com/PaddlePaddle/docs/blob/develop/docs/eval/【Hackathon%20No.113】%20PR.md)
2. [https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/hardware_support/ipu_docs/paddle_install_cn.md](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/hardware_support/ipu_docs/paddle_install_cn.md)
3. [https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/hardware_support/npu_docs/paddle_install_cn.md](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/hardware_support/npu_docs/paddle_install_cn.md)
4. [https://github.com/PaddlePaddle/docs/blob/develop/docs/install/compile/linux-compile.md](https://github.com/PaddlePaddle/docs/blob/develop/docs/install/compile/linux-compile.md)
5. [https://github.com/PaddlePaddle/docs/blob/develop/docs/faq/params_cn.md](https://github.com/PaddlePaddle/docs/blob/develop/docs/faq/params_cn.md)
6. [https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/hardware_support/hardware_info_cn.md](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/hardware_support/hardware_info_cn.md)
7. [https://github.com/PaddlePaddle/docs/blob/develop/docs/dev_guides/docs_contributing_guides_cn.md](https://github.com/PaddlePaddle/docs/blob/develop/docs/dev_guides/docs_contributing_guides_cn.md)
8. [https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.autocast.md](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.autocast.md)
9. [https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.cpu.amp.autocast.md](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.cpu.amp.autocast.md)
小结
可以看到通过引入文档构建的知识库，能够从一定程度上弥补大模型知识上的不足

但是受限于文档的格式、切分方式以及文心一言的 Embedding API 的编码效果，匹配到的文档片段并不是每次都很精准

当然，这只是一个非常简单的知识库引入方式，还有很多可以改进和优化的地方

未来有机会再做更多的介绍，下次一定