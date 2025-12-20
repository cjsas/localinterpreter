import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v3",
    map_location='cpu'
)

output = asr_model.transcribe(['2086-149220-0033.wav'])
print(output[0].text)

