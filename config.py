
class Config:
    use_https = False # 'http' 'https
    port = 8888
    host = '127.0.0.1'
    model_path = '/Users/raojiajun/.cache/modelscope/hub/models/iic/SenseVoiceSmall'
    wss_path = ('wss' if use_https else 'ws') + f'://{host}:{port}'
