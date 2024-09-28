from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
from funasr import AutoModel
import numpy as np
import soundfile as sf
import argparse
import uvicorn
from urllib.parse import parse_qs
import os
import re
from modelscope.pipelines import pipeline
from custom_logger import setup_custom_logger

logger = setup_custom_logger()


class Config(BaseSettings):
    sv_thr: float = Field(0.25, description="Speaker verification threshold")
    chunk_size_ms: int = Field(300, description="Chunk size in milliseconds")
    sample_rate: int = Field(16000, description="Sample rate in Hz")
    bit_depth: int = Field(16, description="Bit depth")
    channels: int = Field(1, description="Number of audio channels")
    avg_logprob_thr: float = Field(-0.2, description="average logprob threshold")

config = Config()

emo_dict = {
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
}

event_dict = {
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|Cry|>": "😭",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "🤧",
}

emoji_dict = {
	"<|nospeech|><|Event_UNK|>": "❓",
	"<|zh|>": "",
	"<|en|>": "",
	"<|yue|>": "",
	"<|ja|>": "",
	"<|ko|>": "",
	"<|nospeech|>": "",
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
	"<|Cry|>": "😭",
	"<|EMO_UNKNOWN|>": "",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "😷",
	"<|Sing|>": "",
	"<|Speech_Noise|>": "",
	"<|withitn|>": "",
	"<|woitn|>": "",
	"<|GBG|>": "",
	"<|Event_UNK|>": "",
}

lang_dict =  {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷",}

def format_str(s):
	for sptk in emoji_dict:
		s = s.replace(sptk, emoji_dict[sptk])
	return s


def format_str_v2(s):
	sptk_dict = {}
	for sptk in emoji_dict:
		sptk_dict[sptk] = s.count(sptk)
		s = s.replace(sptk, "")
	emo = "<|NEUTRAL|>"
	for e in emo_dict:
		if sptk_dict[e] > sptk_dict[emo]:
			emo = e
	for e in event_dict:
		if sptk_dict[e] > 0:
			s = event_dict[e] + s
	s = s + emo_dict[emo]

	for emoji in emo_set.union(event_set):
		s = s.replace(" " + emoji, emoji)
		s = s.replace(emoji + " ", emoji)
	return s.strip()

def format_str_v3(s):
	def get_emo(s):
		return s[-1] if s[-1] in emo_set else None
	def get_event(s):
		return s[0] if s[0] in event_set else None

	s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
	for lang in lang_dict:
		s = s.replace(lang, "<|lang|>")
	s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
	new_s = " " + s_list[0]
	cur_ent_event = get_event(new_s)
	for i in range(1, len(s_list)):
		if len(s_list[i]) == 0:
			continue
		if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
			s_list[i] = s_list[i][1:]
		#else:
		cur_ent_event = get_event(s_list[i])
		if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
			new_s = new_s[:-1]
		new_s += s_list[i].strip().lstrip()
	new_s = new_s.replace("The.", " ")
	return new_s.strip()

def contains_chinese_english_number(s: str) -> bool:
    # Check if the string contains any Chinese character, English letter, or Arabic number
    return bool(re.search(r'[\u4e00-\u9fffA-Za-z0-9]', s))


sv_pipeline = pipeline(
    task='speaker-verification',
    model='iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common',
    model_revision='v1.0.1'
)

model_asr = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    remote_code="./model.py",    
    device="cuda:0",
    disable_update=True
)

model_vad = AutoModel(
    model="fsmn-vad",
    model_revision="v2.0.4",
    disable_pbar = True,
    max_end_silence_time=400,
    speech_noise_thres=0.8,
    disable_update=True,
)

reg_spks_files = [
    "speaker/speaker_tf_16k.wav"
]

def reg_spk_init(files):
    reg_spk = {}
    for f in files:
        data, sr = sf.read(f, dtype="float32")
        k, _ = os.path.splitext(os.path.basename(f))
        reg_spk[k] = {
            "data": data,
            "sr":   sr,
        }
    return reg_spk

reg_spks = reg_spk_init(reg_spks_files)

def speaker_verify(audio, sv_thr):
    hit = False
    for k, v in reg_spks.items():
        res_sv = sv_pipeline([audio, v["data"]], sv_thr)
        logger.info(f"[speaker check] {k}: {res_sv}")
        if res_sv["score"] >= sv_thr:
           hit = True
    return hit, k

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logger.error("Exception occurred", exc_info=True)
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
        message = exc.detail
        data = ""
    elif isinstance(exc, RequestValidationError):
        status_code = HTTP_422_UNPROCESSABLE_ENTITY
        message = "Validation error: " + str(exc.errors())
        data = ""
    else:
        status_code = 500
        message = "Internal server error: " + str(exc)
        data = ""

    return JSONResponse(
        status_code=status_code,
        content=TranscriptionResponse(
            code=status_code,
            msg=message,
            data=data
        ).model_dump()
    )

# Define the response model
class TranscriptionResponse(BaseModel):
    code: int
    msg: str
    data: str

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    
    try:

        query_params = parse_qs(websocket.scope['query_string'].decode())
        sv = query_params.get('sv', ['false'])[0].lower() in ['true', '1', 't', 'y', 'yes']
        lang = query_params.get('lang', ['auto'])[0].lower()
        
        await websocket.accept()
        chunk_size = int(config.chunk_size_ms * config.sample_rate / 1000)
        audio_buffer = np.array([], dtype=np.float32)
        audio_vad = np.array([], dtype=np.float32)
        
        if sv:
            sv_audio_buffer = np.array([], dtype=np.float32)

        cache = {}
        last_vad_beg = last_vad_end = -1
        offset = 0
        hit = False

        while True:
            data = await websocket.receive_bytes()
            #logger.info(f"received {len(data)} bytes")                
            audio_buffer = np.append(audio_buffer, np.frombuffer(data, dtype=np.float32))
                
            while len(audio_buffer) >= chunk_size:
                chunk = audio_buffer[:chunk_size]
                audio_buffer = audio_buffer[chunk_size:]
                
                if last_vad_beg > 1 and sv:
                    # speaker verify
                    # If no hit is detected, continue accumulating audio data and check again until a hit is detected
                    # the `hit`、`sv_audio_buffer` will reset after `model_asr.generate`).
                    if not hit:
                        sv_audio_buffer = np.append(sv_audio_buffer, chunk)
                        hit, speaker = speaker_verify(sv_audio_buffer, config.sv_thr)
                        if hit:
                            response = TranscriptionResponse(
                                code=2,
                                msg="detect speaker",
                                data=speaker
                            )
                            await websocket.send_json(response.model_dump())
                            sv_audio_buffer = np.array([], dtype=np.float32)
                
                audio_vad = np.append(audio_vad, chunk)

                res = model_vad.generate(input=chunk, cache=cache, is_final=False, chunk_size=config.chunk_size_ms)
                #logger.info(f"vad inference: {res}")
                if len(res[0]["value"]):
                    vad_segments = res[0]["value"]
                    for segment in vad_segments:
                        if segment[0] > -1: # speech begin
                            last_vad_beg = segment[0]

                            
                        if segment[1] > -1: # speech end
                            last_vad_end = segment[1]

                        if last_vad_beg > -1 and last_vad_end > -1:
                            logger.info(f"vad segment: {[last_vad_beg, last_vad_end]}")
                            last_vad_beg -= offset
                            last_vad_end -= offset
                            offset += last_vad_end
                            beg = int(last_vad_beg * config.sample_rate / 1000)
                            end = int(last_vad_end * config.sample_rate / 1000)
                            result = model_asr.generate(
                                input       = audio_vad[beg:end],
                                cache       = {},
                                language    = lang.strip(),
                                use_itn     = True,
                                batch_size_s= 60,
                            ) if hit else None
                            logger.info(f"model_asr.generate {result}")
                            audio_vad = audio_vad[end:]
                            last_vad_beg = last_vad_end = -1
                            
                            if sv:
                                # reset `hit`
                                hit = False
                                # reset `sv_audio_buffer`
                                sv_audio_buffer = np.array([], dtype=np.float32)
                            
                            if  result is not None:
                                if result[0]['avg_logprob'] < config.avg_logprob_thr:
                                    data = '...'
                                    logger.info(f'avg_logprob < {config.avg_logprob_thr}, so output ...')
                                else:
                                    data = format_str_v3(result[0]['text'])
                                
                                if data == '...' or contains_chinese_english_number(data):
                                    response = TranscriptionResponse(
                                        code=0,
                                        msg=f"success",
                                        data=data
                                    )
                                    await websocket.send_json(response.model_dump())

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await websocket.close()
    finally:
        audio_buffer = np.array([], dtype=np.float32)
        audio_vad = np.array([], dtype=np.float32)
        sv_audio_buffer = np.array([], dtype=np.float32)
        cache.clear()
        logger.info("Cleaned up resources after WebSocket disconnect")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI app with a specified port.")
    parser.add_argument('--port', type=int, default=27000, help='Port number to run the FastAPI app on.')
    #parser.add_argument('--certfile', type=str, default='/etc/perm/labs.makee.com_bundle.crt', help='SSL certificate file')
    #parser.add_argument('--keyfile', type=str, default='/etc/perm/labs.makee.com.key', help='SSL key file')

    args = parser.parse_args()
    
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn": {  # uvicorn logger
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }
    
    #uvicorn.run(app, host="0.0.0.0", port=args.port, ssl_certfile=args.certfile, ssl_keyfile=args.keyfile)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_config=log_config)