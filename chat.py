import erniebot

erniebot.api_type = 'aistudio'
erniebot.access_token = '19d47a3bf99d0c29a3437b83e567e5bef74659c2'


models = erniebot.Model.list()

print(models)

response = erniebot.ChatCompletion.create(
    model='ernie-bot-4',
    messages=[{
                'role': 'user',
                'content': '你好，请介绍下你自己？'
    }],
    stream=True
)

for chunk in response:
    print(chunk['result'], end='')