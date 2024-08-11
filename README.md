API and websocket server for sensevoice

It has inherited some enhanced features, such as VAD detection, real-time streaming recognition, and speaker verification.

## 安装

首先，克隆此仓库到你的本地机器：

```bash
git clone https://github.com/0x5446/api4sensevoice.git
cd api4sensevoice
```

然后，使用以下命令安装所需的依赖项：

```bash
pip install -r requirements.txt
```

## 运行

### 一句话识别API Server

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI app with a specified port.")
    parser.add_argument('--port', type=int, default=7000, help='Port number to run the FastAPI app on.')
    parser.add_argument('--certfile', type=str, default='path_to_your_certfile', help='SSL certificate file')
    parser.add_argument('--keyfile', type=str, default='path_to_your_keyfile', help='SSL key file')
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port, ssl_certfile=args.certfile, ssl_keyfile=args.keyfile)
```
以上是server.py结尾部分的代码，可以自行修改来定义port、certfile、keyfile，然后直接运行python server.py启动API服务。

也可以通过启动命令参数来设置，比如：

```bash
python server.py --port 8888 --certfile /etc/perm/your_domain_bundle.crt --keyfile /etc/perm/your_domain.key
```

#### API 说明

##### Transcribe Audio

- **路径**: `/transcribe`
- **方法**: `POST`
- **概要**: 转录音频
- **请求体**:
  - `multipart/form-data`
  - **参数**:
    - `file` (必须): 要转录的音频文件

- **响应**:
  - **200 成功响应**:
    - **内容类型**: `application/json`
    - **Schema**:
      - `errno` (integer): 错误编号
      - `errmsg` (string): 错误信息
      - `resp` (object): 响应对象
  - **422 验证错误**:
    - **内容类型**: `application/json`
    - **Schema**:
      - `detail` (array): 错误详情数组

- **请求示例**:

```bash
curl -X 'POST'  
  'http://yourapiaddress/transcribe'  
  -H 'accept: application/json'  
  -H 'Content-Type: multipart/form-data'  
  -F 'file=@path_to_your_audio_file'
```

- **响应示例**（200 成功响应）:

```json
{
  "code": 0,
  "msg": "Success",
  "data": {
    // 转录结果
  }
}
```


### 流式实时识别WebSocket Server

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI app with a specified port.")
    parser.add_argument('--port', type=int, default=27000, help='Port number to run the FastAPI app on.')
    parser.add_argument('--certfile', type=str, default='path_to_your_certfile', help='SSL certificate file')
    parser.add_argument('--keyfile', type=str, default='path_to_your_keyfile', help='SSL key file')
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port, ssl_certfile=args.certfile, ssl_keyfile=args.keyfile)
```
以上是server_wss.py结尾部分的代码，可以自行修改来定义port、certfile、keyfile，然后直接运行python server_wss.py启动WebSocket服务。

也可以通过启动命令参数来设置，比如：

```bash
python server_wss.py --port 8888 --certfile /etc/perm/your_domain_bundle.crt --keyfile /etc/perm/your_domain.key
```

如果要开启说话人验证：
1. 准备需要验证的说话人的人声音频文件：16000采样率，单通道，16位宽，wav格式，放到speaker目录下。
2. 请修改server_wss.py如下部分，将list里的文件路径替换成自己的（可以添加多个，命中任何一个均为校验通过，会进行asr推理）
```python
reg_spks_files = [
    "speaker/speaker1_a_cn_16k.wav"
]
```

#### WebSocket参数
- 端点：/ws/transcribe
- Query参数：
  - sv：是否开启说话人验证
    - 可选
    - 默认值：0
- 上行数据：PCM二进制
- 下行数据：String
  - **Schema**:
    - `errno` (integer): 错误编号
    - `errmsg` (string): 错误信息
    - `resp` (object): 响应对象

#### client测试页面
- client_wss.html
- 把wsUrl修改为自己websocket服务的地址即可测试
```javascript
const wsUrl = `wss://127.0.0.1:27000/ws/transcribe?sv=${svEnabled}`; // change to your websocket server address
```

## 计划

- [x]  一句话识别（适合一小段语音）
- [x]  流式实时识别
- [x]  流式实时识别支持说话人验证
- [ ]  延时优化

## 贡献

欢迎所有形式的贡献，包括但不限于：

- 报告Bug
- 提出功能请求
- 提交代码改进
- 更新文档

## License

此项目遵循[MIT License](https://opensource.org/license/mit)。详情请参阅LICENSE文件。

## 依赖项目
[https://github.com/FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
[https://modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced](https://modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced)
[https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)