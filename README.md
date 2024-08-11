# SenseVoice API Server

这是为SenseVoice编写的API Server，旨在为该项目提供API服务接口。

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

要运行API Server，请使用以下命令：

```bash
python api.py
```

## API 说明

### Transcribe Audio

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

请求示例:

```bash
curl -X 'POST'  
  'http://yourapiaddress/transcribe'  
  -H 'accept: application/json'  
  -H 'Content-Type: multipart/form-data'  
  -F 'file=@path_to_your_audio_file'
```

响应示例（200 成功响应）:

```json
{
  "errno": 0,
  "errmsg": "Success",
  "resp": {
    // 转录结果
  }
}
```

## 计划

- [x]  一句话识别（适合一小段语音）
- [ ]  流式实时识别

## 贡献

欢迎所有形式的贡献，包括但不限于：

- 报告Bug
- 提出功能请求
- 提交代码改进
- 更新文档

## License

此项目遵循[MIT License](https://opensource.org/license/mit)。详情请参阅LICENSE文件。

## 更多信息

其他说明参见SenseVoice项目主页：
[https://github.com/FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
