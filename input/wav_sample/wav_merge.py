from pydub import AudioSegment

# WAVファイルを読み込み
wav1 = AudioSegment.from_wav("jpn_inoue_16k.wav")
wav2 = AudioSegment.from_wav("jpn_sumida_16k.wav")

# 合成する（重ねる）
combined = wav1.overlay(wav2)

# ファイルとして保存
combined.export("jpn_inoue_sumida_16k.wav", format="wav")