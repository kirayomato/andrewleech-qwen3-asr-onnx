from asr_model import Qwen3ASRModel
import librosa

m = Qwen3ASRModel(
    r"D:\Program\Github\Qwen3-ASR-onnx\andrewleech\qwen3-asr-0.6b-onnx-int4-merged",
    context="微分",
    language="en",
)
m.load()
wav, _ = librosa.load(r"D:\Program\Github\Qwen3-ASR-onnx\test_wavs\rag_math.wav")
print(m.transcribe([wav]))  # 应与 split 版逐字相同
